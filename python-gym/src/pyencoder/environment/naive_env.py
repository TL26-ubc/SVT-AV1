import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import av
import cv2
import gymnasium as gym
import numpy as np
from pyencoder.environment.av1_runner import Av1Runner
from pyencoder.utils.video_reader import VideoReader
from pyencoder.environment.constants import SB_SIZE, QP_MIN, QP_MAX
import math
from pyencoder.states.abstract import AbstractState
from pyencoder.states.naive import NaiveState

# Extending gymnasium's Env class
# https://gymnasium.farama.org/api/env/#gymnasium.Env
class Av1GymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        lambda_rd: float = 0.1,
        queue_timeout=10,
        state: type[AbstractState] = NaiveState,
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.av1_runner = Av1Runner(video_path)

        # Initialize the VideoReader
        self.video_reader = VideoReader(path=video_path)

        self.lambda_rd = lambda_rd
        self._episode_done = threading.Event()

        self.num_superblocks = self.video_reader.get_num_superblock()
        self.num_frames = self.video_reader.get_frame_count()

        # Action space = QP offset grid
        self.action_space = gym.spaces.MultiDiscrete(
            # num \in [QP_MAX, QP_MIN], there are num_superblocks of them
            [QP_MAX - QP_MIN + 1]
            * self.num_superblocks
        )

        # RL/encoder communication
        self.queue_timeout = queue_timeout

        # Episode management
        self.current_frame = 0
        self.terminated = False

        # Frame data storage
        self.frame_history = []
        self.reward_history = []  # For temporal attribution
        
        # Target bitrate for reward calculation
        self.target_bitrate = 600 * 1000 / 8  # Convert from kbps to bytes
        
        # Reward scaling parameters
        self.psnr_mean = 35.0
        self.psnr_std = 15.0
        self.ssim_mean = 0.5
        self.ssim_std = 0.5

        # Synchronization
        self.encoder_thread = None

        # run the first round of encoding, save the baseline video at specified output path
        self.av1_runner.run(
            output_path=f"{str(output_dir)}/baseline_output.ivf"
        )

        # Get baseline observations
        baseline_obs = []
        for _ in range(self.num_frames):
            obs = self.av1_runner.wait_for_next_observation()
            baseline_obs.append(obs)
            self.av1_runner.send_action_response(skip=True)
            _ = self.av1_runner.wait_for_feedback()

        # ensure thread has finished
        self.av1_runner.join()

        self.state_wrapper = state(video_reader=self.video_reader, baseline_observations=baseline_obs, sb_size=SB_SIZE)

        self.observation_space = self.state_wrapper.get_observation_space()

        self.y_psnr_list = []
        self.cb_psnr_list = []
        self.cr_psnr_list = []
        self.baseline_heighest_psnr = {
            "y": -114514.0,  # Initialize with very low values
            "cb": -114514.0,
            "cr": -114514.0,
        }
        
        # Save baseline frame PSNRs
        self.save_baseline_frame_psnr(
            baseline_video_path=f"{str(output_dir)}/baseline_output.ivf"
        )
        
    def save_baseline_frame_psnr(self, baseline_video_path: str | Path):
        """Calculate and save PSNR for baseline frames"""
        baseline_video_path = str(baseline_video_path)
        container = av.open(baseline_video_path)
        stream = container.streams.video[0]

        total_frames = int(stream.frames)
        assert (
            total_frames == self.num_frames
        ), f"Baseline video frame count {total_frames} does not match expected {self.num_frames}"

        assert (
            self.y_psnr_list == []
        ), "PSNR lists should be empty before saving baseline PSNRs"
        assert (
            self.cb_psnr_list == []
        ), "PSNR lists should be empty before saving baseline PSNRs"
        assert (
            self.cr_psnr_list == []
        ), "PSNR lists should be empty before saving baseline PSNRs"

        for frame_number, frame in enumerate(container.decode(stream)):
            # Convert frame to YCbCr and get numpy arrays for Y, Cb, Cr
            ycbcr = frame.to_ndarray(format="yuv420p") # (3/2 * H, W)

            # Get YCbCr PSNR for the current frame
            y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(frame_number, ycbcr, self.baseline_heighest_psnr)            
            # Validate PSNR values
            if not np.all(np.isfinite([y_psnr, cb_psnr, cr_psnr])):
                invalid_names = [name for val, name in zip([y_psnr, cb_psnr, cr_psnr], ["Y", "Cb", "Cr"]) if not np.isfinite(val)]
                print(f"Warning: Invalid PSNR(s) {invalid_names} for baseline frame {frame_number}")
                raise InvalidStateError(
                    f"Invalid PSNR(s) {invalid_names} for baseline frame {frame_number}"
                )
            
            # Append to lists
            self.y_psnr_list.append(y_psnr)
            self.cb_psnr_list.append(cb_psnr)
            self.cr_psnr_list.append(cr_psnr)

        assert (
            len(self.y_psnr_list) == self.num_frames
        ), f"Expected {self.num_frames} Y PSNR values, got {len(self.y_psnr_list)}"
        assert (
            len(self.cb_psnr_list) == self.num_frames
        ), f"Expected {self.num_frames} Cb PSNR values, got {len(self.cb_psnr_list)}"
        assert (
            len(self.cr_psnr_list) == self.num_frames
        ), f"Expected {self.num_frames} Cr PSNR values, got {len(self.cr_psnr_list)}"
        container.close()
        
        self.baseline_heighest_psnr['y'] = max(self.y_psnr_list)
        self.baseline_heighest_psnr['cb'] = max(self.cb_psnr_list)
        self.baseline_heighest_psnr['cr'] = max(self.cr_psnr_list)
        
        # iterate through each list, replace negative values with heighest PSNR
        for i in range(self.num_frames):
            if self.y_psnr_list[i] < 0:
                self.y_psnr_list[i] = self.baseline_heighest_psnr['y']
            if self.cb_psnr_list[i] < 0:
                self.cb_psnr_list[i] = self.baseline_heighest_psnr['cb']
            if self.cr_psnr_list[i] < 0:
                self.cr_psnr_list[i] = self.baseline_heighest_psnr['cr']
        
        print("Baseline frame PSNRs saved:")

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[dict, dict]:
        print("Resetting environment...")

        super().reset(seed=seed)

        # Reset episode state
        self.current_frame = 0
        self.current_episode_reward = 0.0
        self.terminated = False
        self.frame_history.clear()
        self.reward_history.clear()

        # Start encoder in separate thread
        self.av1_runner.run()

        # Get initial observation
        initial_obs = self._get_next_observation()

        info = {
            "frame_number": self.current_frame,
            "reward": 0,
            "bitstream_size": 0,
            "episode_frames": self.current_frame,
        }

        return initial_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.terminated:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        # # Validate action
        # if action.shape != (self.num_superblocks,):
        #     raise ValueError(
        #         f"Action shape {action.shape} != expected ({self.num_superblocks},)"
        #     )
          
        # # Validate action values
        # validate_array(action, f"Action for frame {self.current_frame}")
        
        # Convert action to QP offsets
        qp_offsets = self._action_to_qp_offsets(action)
        
        # Send action response to encoder
        self.av1_runner.send_action_response(action=qp_offsets.tolist())
        
        # Wait for encoding feedback
        feedback = self.av1_runner.wait_for_feedback()

        # Compute quality metrics FIRST for state history update
        encoded_frame_data = feedback["encoded_frame_data"]
        current_bitstream_size = feedback.get("bitstream_size", 0)
        
        # Get YCbCr PSNR and SSIM for the current frame
        y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(
            self.current_frame, encoded_frame_data, self.baseline_heighest_psnr
        )
        y_ssim, cb_ssim, cr_ssim = self.video_reader.ycrcb_ssim(
            self.current_frame, encoded_frame_data
        )
        
        # Update state history BEFORE reward calculation so reward can use historical context
        if hasattr(self.state_wrapper, 'update_history'):
            quality_metrics = {
                'y_psnr': y_psnr,
                'cb_psnr': cb_psnr,
                'cr_psnr': cr_psnr,
                'y_ssim': y_ssim,
                'cb_ssim': cb_ssim,
                'cr_ssim': cr_ssim
            }
            self.state_wrapper.update_history(self.current_frame, quality_metrics, current_bitstream_size)

        # Calculate reward using already computed quality metrics
        reward = self._calculate_reward_with_metrics(
            y_psnr, cb_psnr, cr_psnr, y_ssim, cb_ssim, cr_ssim, current_bitstream_size
        )
        self.current_episode_reward += reward

        # Store reward for temporal attribution
        self.reward_history.append(reward)
        
        # Update episode state
        self.current_frame += 1

        # Check termination conditions
        self._check_termination_conditions()
        
        # Get next observation with validation
        if self.terminated:
            # Episode has ended, return dummy observation
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            print(f"Episode terminated at frame {self.current_frame-1} (total frames: {self.num_frames})")
        else:
            # Get next observation with validation
            try:
                next_obs = self._get_next_observation()
            except Exception as e:
                print(f"Failed to get observation for frame {self.current_frame}: {e}")
                # Force termination and return dummy observation
                self.terminated = True
                next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Store frame data with enhanced metrics
        frame_data = {
            "frame_number": self.current_frame,
            "qp_offsets": qp_offsets,
            "reward": reward,
            "feedback": feedback,
        }
        
        # Add quality metrics for analysis
        if hasattr(self, 'current_frame_psnr'):
            frame_data["quality_metrics"] = self.current_frame_psnr
        
        self.frame_history.append(frame_data)

        info = {
            "frame_number": self.current_frame,
            "reward": reward,
            "bitstream_size": feedback.get("bitstream_size", 0),
            "episode_frames": self.current_frame,
        }
        
        # Add PSNR values to info if available
        if hasattr(self, 'current_frame_psnr'):
            info.update(self.current_frame_psnr)
            
        # Add diagnostic metrics
        if len(self.reward_history) > 0:
            info["reward_variance"] = np.var(self.reward_history)
            info["reward_mean"] = np.mean(self.reward_history)
            info["reward_std"] = np.std(self.reward_history)

        return next_obs, reward, self.terminated, False, info

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.close
    def close(self):
        print("Closing environment...")

        if self.encoder_thread and self.encoder_thread.is_alive():
            self._episode_done.set()
            self.encoder_thread.join(timeout=2.0)
        
        print("Environment closed")

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    def render(self):
        pass
    
    def save_bitstream_to_file(self, output_path: str, interrupt: bool = False):
        """Save the bitstream to a file"""
        self.av1_runner.save_bitstream_to_file(output_path, interrupt=interrupt)

    def _get_next_observation(self) -> np.ndarray:
        """Get current observation based on current frame"""
        try:
            observation = self.av1_runner.wait_for_next_observation()
            current_frame = self.video_reader.read_frame(frame_number=observation.picture_number)

            # Assert frame numbers match
            assert (
                observation.picture_number == self.current_frame
            ), f"observation frame {observation.picture_number} != current_frame {self.current_frame}"

            frame_state = self.state_wrapper.get_observation(
                current_frame, 
                observation.superblocks, 
                observation.frame_type, 
                observation.picture_number
            )
            
            # Validate the observation
            validate_array(frame_state, f"Current observation (frame {self.current_frame})")
            
            return frame_state
            
        except Exception as e:
            raise InvalidStateError(f"Failed to get current observation for frame {self.current_frame}: {e}")

    def _action_to_qp_offsets(self, action: np.ndarray) -> np.ndarray:
        """Convert discrete action to QP offsets"""
        # Map from [0, QP_MAX-QP_MIN] to [QP_MIN, QP_MAX]
        qp_offsets = action + QP_MIN
        qp_offsets = np.clip(qp_offsets, QP_MIN, QP_MAX)
        
        # # Validate QP offsets
        # validate_array(qp_offsets, f"QP offsets for frame {self.current_frame}")
        
        return qp_offsets

    def _calculate_reward(
        self, feedback: Dict, qp_offsets: np.ndarray
    ) -> float:
        """Calculate enhanced reward with proper scaling and temporal attribution"""
        # qp_offsets parameter kept for future QP-based reward features
        try:
            # Get encoded frame data and bitstream size
            encoded_frame_data = feedback["encoded_frame_data"]
            current_bitstream_size = feedback.get("bitstream_size", 0)
            
            # Get YCbCr PSNR for the current frame
            y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(
                self.current_frame, encoded_frame_data, self.baseline_heighest_psnr
            )
            
            # Get YCbCr SSIM for the current frame
            y_ssim, cb_ssim, cr_ssim = self.video_reader.ycrcb_ssim(
                self.current_frame, encoded_frame_data
            )
            
            # Apply proper reward scaling normalization
            reward = self._normalize_reward(y_psnr, y_ssim, current_bitstream_size)
            
            # Apply temporal attribution if we have enough history
            if len(self.reward_history) >= 1:
                reward = self._apply_temporal_attribution(reward)
            
            self.current_frame_psnr = {
                'y_psnr': y_psnr,
                'cb_psnr': cb_psnr,
                'cr_psnr': cr_psnr,
                'y_ssim': y_ssim,
                'cb_ssim': cb_ssim,
                'cr_ssim': cr_ssim,
                'bitstream_size': current_bitstream_size,
                'total_reward': reward
            }
            
            return float(reward)
            
        except Exception as e:
            print(f"Reward calculation failed for frame {self.current_frame}: {e}")
            return 0.0
    
    def _normalize_reward(self, psnr: float, ssim: float, bitrate: int) -> float:
        """Normalize reward components to [-1, +1] range for neural network stability"""
        # Normalize PSNR to [-1, +1] range (typical range 20-50 dB)
        psnr_norm = (psnr - self.psnr_mean) / self.psnr_std
        psnr_norm = np.clip(psnr_norm, -1.0, 1.0)
        
        # Normalize SSIM to [-1, +1] range (already 0-1, shift to -1,+1)
        ssim_norm = (ssim - self.ssim_mean) / self.ssim_std
        ssim_norm = np.clip(ssim_norm, -1.0, 1.0)
        
        # Bitrate penalty normalized to target
        bitrate_penalty = -abs(bitrate - self.target_bitrate) / self.target_bitrate
        bitrate_penalty = np.clip(bitrate_penalty, -1.0, 0.0)
        
        # Weighted combination in proper range [-3, +3]
        reward = 0.4 * psnr_norm + 0.4 * ssim_norm + 0.2 * bitrate_penalty
        
        return np.clip(reward, -3.0, 3.0)
    
    def _apply_temporal_attribution(self, current_reward: float, gamma: float = 0.95, n_steps: int = 4) -> float:
        """Apply n-step temporal attribution for delayed QP effects"""
        # Get recent rewards for temporal attribution
        recent_rewards = self.reward_history[-min(n_steps, len(self.reward_history)):]
        
        # Calculate temporal reward with exponential decay
        temporal_reward = current_reward
        for i, past_reward in enumerate(reversed(recent_rewards)):
            temporal_reward += (gamma ** (i + 1)) * past_reward
        
        # Normalize by number of steps to prevent reward inflation
        temporal_reward /= (1 + len(recent_rewards))
        
        return temporal_reward
    
    def _calculate_reward_with_metrics(
        self, y_psnr: float, cb_psnr: float, cr_psnr: float, 
        y_ssim: float, cb_ssim: float, cr_ssim: float, current_bitstream_size: int
    ) -> float:
        """Calculate reward using pre-computed quality metrics"""
        try:
            # Apply proper reward scaling normalization
            reward = self._normalize_reward(y_psnr, y_ssim, current_bitstream_size)
            
            # Apply temporal attribution if we have enough history
            if len(self.reward_history) >= 1:
                reward = self._apply_temporal_attribution(reward)
            
            # Store current frame metrics for info
            self.current_frame_psnr = {
                'y_psnr': y_psnr,
                'cb_psnr': cb_psnr,
                'cr_psnr': cr_psnr,
                'y_ssim': y_ssim,
                'cb_ssim': cb_ssim,
                'cr_ssim': cr_ssim,
                'bitstream_size': current_bitstream_size,
                'total_reward': reward
            }
            
            return float(reward)
            
        except Exception as e:
            print(f"Reward calculation failed for frame {self.current_frame}: {e}")
            return 0.0
    
    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Maximum frames reached
        if self.current_frame >= self.num_frames:
            self.terminated = True
            
        # Early termination for extremely poor performance
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-10:]
            if np.mean(recent_rewards) < -2.0:  # Consistently very poor rewards
                print(f"Early termination due to poor performance: mean reward = {np.mean(recent_rewards):.3f}")
                self.terminated = True

class InvalidStateError(Exception):
    pass


class InvalidRewardError(Exception):
    pass


def validate_array(arr: np.ndarray, name: str) -> None:
    """Validate numpy array for NaN, inf, and other invalid values"""
    if arr is None:
        raise ValueError(f"{name} is None")

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(arr)}")

    if arr.size == 0:
        raise ValueError(f"{name} is empty")

    # Check for NaN values
    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        nan_indices = np.where(nan_mask)
        raise InvalidStateError(
            (
                (
                    f"{name} contains {nan_count} NaN values at indices: {nan_indices}. "
                    f"Array shape: {arr.shape}, dtype: {arr.dtype}\n"
                    f"Array sample: {arr.flat[:min(10, arr.size)]}"
                )
            )
        )

    # Check for infinite values
    inf_mask = np.isinf(arr)
    if np.any(inf_mask):
        inf_count = np.sum(inf_mask)
        inf_indices = np.where(inf_mask)
        raise InvalidStateError(
            f"{name} contains {inf_count} infinite values at indices: {inf_indices}. "
            f"Array shape: {arr.shape}, dtype: {arr.dtype}\n"
            f"Array sample: {arr.flat[:min(10, arr.size)]}"
        )

    # Check for extremely large values that might cause numerical issues
    max_val = np.max(np.abs(arr))
    if max_val > 1e6:
        print(f"Warning: {name} contains very large values (max abs: {max_val:.2e})")


def validate_reward(
    reward: float,
    frame_number: int,
    details: dict = None
) -> None:
    """Validate reward value"""
    if reward is None:
        raise InvalidRewardError(f"Reward is None for frame {frame_number}")
    
    if not isinstance(reward, (int, float, np.number)):
        raise InvalidRewardError(
            f"Reward must be numeric, got {type(reward)} for frame {frame_number}"
        )
    
    if math.isnan(reward):
        raise InvalidRewardError(
            f"Reward is NaN for frame {frame_number}. Details: {details}"
        )
    
    if math.isinf(reward):
        raise InvalidRewardError(
            f"Reward is infinite ({reward}) for frame {frame_number}. Details: {details}"
        )
    
    # Check for extremely large rewards that might indicate calculation errors
    if abs(reward) > 1000:
        print(f"Warning: Very large reward ({reward:.2f}) for frame {frame_number}")
