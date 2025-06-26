import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import av
import cv2
import gymnasium as gym
import numpy as np
from pyencoder.environment.av1_runner import Av1Runner
from pyencoder.utils.video_reader import VideoReader
from sympy import false
import math

# Constants
QP_MIN, QP_MAX = -3, 3  # delta QP range which will be action
SB_SIZE = 64  # superblock size


# @dataclass()
# class Av1GymInfo:
#     actions: np.ndarray
#     reward: float

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
        queue_timeout=10,  # timeout
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.av1_runner = Av1Runner(video_path=video_path)
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

        # Enhanced observation space with history
        # Features per superblock: current + 2 previous frames + global info
        self.history_length = 3  # Current + 2 previous frames
        
        # Per-superblock features (4 features * 3 frames = 12 features per block)
        self.features_per_block = 4 * self.history_length
        
        # Global features: frame-level statistics and history
        self.global_features = 16  # Global video and encoding statistics
        
        total_features = self.features_per_block * self.num_superblocks + self.global_features
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            # Enhanced features:
            # Per superblock (12 features * num_superblocks):
            #   Current frame: Y-variance, H-motion, V-motion, gradient
            #   Frame t-1: Y-variance, H-motion, V-motion, gradient  
            #   Frame t-2: Y-variance, H-motion, V-motion, gradient
            # Global features (16 features):
            #   - Frame position in video (normalized)
            #   - Previous 3 frame rewards
            #   - Previous 3 average QP decisions
            #   - Scene complexity metrics
            #   - Motion activity level
            #   - Temporal consistency
            #   - Rate control info
            dtype=np.float32,
        )
        
        # History storage for enhanced observations
        self.frame_history_obs = []  # Store frame observations
        self.reward_history = []     # Store recent rewards  
        self.qp_history = []         # Store recent QP decisions
        self.scene_stats = {         # Global scene statistics
            'avg_motion': 0.0,
            'complexity': 0.0,
            'temporal_activity': 0.0
        }

        # RL/encoder communication
        self.queue_timeout = queue_timeout

        # Episode management
        self.current_frame = 0
        self.terminated = False

        # Frame data storage
        self.frame_history = []

        # Synchronization
        self.encoder_thread = None

        # run the first round of encoding, save the baseline video at specified output path
        self.av1_runner.run_SVT_AV1_encoder(
            output_path=f"{str(output_dir)}/baseline_output.ivf"
        )
        self.y_psnr_list = []
        self.cb_psnr_list = []
        self.cr_psnr_list = []
        self.baseline_heighest_psnr = {
            "y": -114514.0,  # Initialize with very low values
            "cb": -114514.0,
            "cr": -114514.0,
        }

        self.save_baseline_frame_psnr(
            baseline_video_path=f"{str(output_dir)}/baseline_output.ivf"
        )

        viz_output_dir = str(output_dir / "observation_analysis")
        self.observation_max_values = (
            self.video_reader.collect_baseline_observation_stats(
                viz_output_dir
            )
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
            ycbcr = frame.to_ndarray(format="rgb24")
            ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_RGB2YCrCb)

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
        super().reset(seed=seed)

        print("Resetting environment...")

        # Reset episode state
        self.current_frame = 0
        self.terminated = False
        self.frame_history.clear()
        
        # Reset enhanced observation history
        self.frame_history_obs.clear()
        self.reward_history.clear()
        self.qp_history.clear()
        self.scene_stats = {
            'avg_motion': 0.0,
            'complexity': 0.0,
            'temporal_activity': 0.0
        }

        # Get initial observation (frame 0 state)
        initial_obs = self._get_initial_observation()

        # Start encoder in separate thread
        self._start_encoder_thread()

        info = {
            "frame_number": self.current_frame,
            "reward": 0,
            "bitstream_size": 0,
            "episode_frames": self.current_frame,
        }

        return initial_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        if self.terminated:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        # Validate action
        if action.shape != (self.num_superblocks,):
            raise ValueError(
                f"Action shape {action.shape} != expected ({self.num_superblocks},)"
            )
          
        # Validate action values
        validate_array(action, f"Action for frame {self.current_frame}")
        
        # Convert action to QP offsets
        qp_offsets = self._action_to_qp_offsets(action)

        # Wait for encoder to request action for this frame
        action_request = self.av1_runner.wait_for_action_request(
            timeout=self.queue_timeout
        )

        if action_request is None:
            print("No action request received - episode terminated")
            self.terminated = True
            dummy_obs = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )
            return dummy_obs, 0.0, True, False, {"timeout": True}
        
        frame_number = action_request["picture_number"]
        
        # Send action response to encoder
        self.av1_runner.send_action_response(qp_offsets.tolist())
        
        # Wait for encoding feedback
        feedback = self.av1_runner.wait_for_feedback(timeout=self.queue_timeout)
        
        if feedback is None:
            print("No feedback received - episode terminated")
            self.terminated = True
            return (
                self._get_current_observation(),
                0.0,
                True,
                False,
                {"no_feedback": True},
            )

        # Calculate reward (return per-frame reward directly, let PPO handle advantages)
        reward = self._calculate_reward(feedback, action_request, qp_offsets)
        
        # Update history for enhanced observations
        self.reward_history.append(reward)
        self.qp_history.append(qp_offsets.copy())
        
        # Update scene statistics  
        if len(self.frame_history_obs) > 0:
            current_frame_obs = self.frame_history_obs[-1]
            if current_frame_obs.shape == (4, self.num_superblocks):
                # Update motion and complexity statistics
                motion_activity = np.mean(np.abs(current_frame_obs[1:3, :]))  # Average motion
                variance_activity = np.mean(current_frame_obs[0, :])  # Average variance
                
                # Update rolling averages
                alpha = 0.1  # Learning rate for rolling average
                self.scene_stats['avg_motion'] = (1 - alpha) * self.scene_stats['avg_motion'] + alpha * motion_activity
                self.scene_stats['complexity'] = (1 - alpha) * self.scene_stats['complexity'] + alpha * variance_activity
                
                # Temporal activity based on frame differences
                if len(self.frame_history_obs) >= 2:
                    prev_frame = self.frame_history_obs[-2]
                    if prev_frame.shape == current_frame_obs.shape:
                        temporal_diff = np.mean(np.abs(current_frame_obs - prev_frame))
                        self.scene_stats['temporal_activity'] = (1 - alpha) * self.scene_stats['temporal_activity'] + alpha * temporal_diff
        
        # Limit history size to prevent memory issues
        max_history = 100
        if len(self.frame_history_obs) > max_history:
            self.frame_history_obs = self.frame_history_obs[-max_history:]
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]
        if len(self.qp_history) > max_history:
            self.qp_history = self.qp_history[-max_history:]

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
                next_obs = self._get_current_observation()
            except Exception as e:
                print(f"Failed to get observation for frame {self.current_frame}: {e}")
                # Force termination and return dummy observation
                self.terminated = True
                next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    
        
        # Store frame data
        self.frame_history.append(
            {
                "frame_number": frame_number,
                "qp_offsets": qp_offsets,
                "reward": reward,
                "feedback": feedback,
            }
        )

        info = {
            "frame_number": frame_number,
            "reward": reward,
            "bitstream_size": feedback.get("bitstream_size", 0),
            "episode_frames": self.current_frame,
        }

        # (f"Step completed - Frame: {frame_number}, Reward: {reward:.4f}")

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

    def _get_initial_observation(self) -> np.ndarray:
        """Get initial enhanced observation for episode start"""
        try:
            # Get current frame observation
            frame_state = self.video_reader.get_x_frame_state_normalized(frame_number=0)
            current_obs = np.array(frame_state, dtype=np.float32)
            
            # Store in history for future use
            self.frame_history_obs.append(current_obs)
            
            # Build enhanced observation with history
            enhanced_obs = self._build_enhanced_observation(current_obs)
            
            return enhanced_obs
            
        except Exception as e:
            raise InvalidStateError(f"Failed to get initial observation: {e}")
    
    def get_observation_stats(self) -> dict:
        if self.observation_max_values is None:
            raise ValueError(
                "Observation normalization is not enabled."
            )
            
        return {
            "max_values": self.observation_max_values,
            "num_superblocks": self.num_superblocks,
            "num_frames": self.num_frames
        }

    def _get_current_observation(self) -> np.ndarray:
        """Get current enhanced observation based on current frame"""
        try:
            # Get current frame observation
            frame_state = self.video_reader.get_x_frame_state_normalized(
                frame_number=self.current_frame
            )
            current_obs = np.array(frame_state, dtype=np.float32)
            
            # Store in history for future use  
            self.frame_history_obs.append(current_obs)
            
            # Build enhanced observation with history
            enhanced_obs = self._build_enhanced_observation(current_obs)
            
            return enhanced_obs
            
        except Exception as e:
            raise InvalidStateError(f"Failed to get current observation for frame {self.current_frame}: {e}")
            
    def _build_enhanced_observation(self, current_obs: np.ndarray) -> np.ndarray:
        """Build enhanced observation with temporal history and global features"""
        try:
            # Prepare per-superblock features with history
            per_block_features = []
            
            # Get history frames (current + up to 2 previous)
            history_frames = self.frame_history_obs[-self.history_length:]
            
            # Pad with zeros if we don't have enough history yet
            while len(history_frames) < self.history_length:
                zero_frame = np.zeros_like(current_obs)
                history_frames.insert(0, zero_frame)
            
            # Flatten per-superblock features across time
            for sb_idx in range(self.num_superblocks):
                sb_features = []
                for frame_obs in history_frames:
                    # Extract 4 features for this superblock from this frame
                    if frame_obs.shape == (4, self.num_superblocks):
                        sb_features.extend(frame_obs[:, sb_idx])
                    else:
                        # Handle empty frames
                        sb_features.extend([0.0, 0.0, 0.0, 0.0])
                per_block_features.extend(sb_features)
            
            # Build global features
            global_features = self._build_global_features()
            
            # Combine all features
            enhanced_obs = np.array(per_block_features + global_features, dtype=np.float32)
            
            # Validate final observation
            validate_array(enhanced_obs, f"Enhanced observation (frame {self.current_frame})")
            
            return enhanced_obs
            
        except Exception as e:
            raise InvalidStateError(f"Failed to build enhanced observation: {e}")
    
    def _build_global_features(self) -> list:
        """Build global/context features for the observation"""
        global_features = []
        
        # 1. Frame position in video (normalized)
        global_features.append(self.current_frame / max(1, self.num_frames - 1))
        
        # 2-4. Previous 3 frame rewards (padded with 0 if not available)
        recent_rewards = self.reward_history[-3:] if self.reward_history else []
        while len(recent_rewards) < 3:
            recent_rewards.insert(0, 0.0)
        global_features.extend(recent_rewards)
        
        # 5-7. Previous 3 average QP decisions (padded with 0 if not available)
        recent_qps = []
        if self.qp_history:
            for qp_frame in self.qp_history[-3:]:
                avg_qp = np.mean(qp_frame) if len(qp_frame) > 0 else 0.0
                recent_qps.append(avg_qp)
        while len(recent_qps) < 3:
            recent_qps.insert(0, 0.0)
        global_features.extend(recent_qps)
        
        # 8-10. Scene complexity metrics
        global_features.append(self.scene_stats['complexity'])
        global_features.append(self.scene_stats['avg_motion'])
        global_features.append(self.scene_stats['temporal_activity'])
        
        # 11-13. Motion activity indicators (based on current frame if available)
        if len(self.frame_history_obs) > 0:
            current_frame_obs = self.frame_history_obs[-1]
            if current_frame_obs.shape == (4, self.num_superblocks):
                # Motion statistics from current frame
                h_motion_std = np.std(current_frame_obs[1, :])  # H-motion variability
                v_motion_std = np.std(current_frame_obs[2, :])  # V-motion variability  
                variance_mean = np.mean(current_frame_obs[0, :])  # Average Y-variance
            else:
                h_motion_std = v_motion_std = variance_mean = 0.0
        else:
            h_motion_std = v_motion_std = variance_mean = 0.0
            
        global_features.extend([h_motion_std, v_motion_std, variance_mean])
        
        # 14-16. Rate control and encoding context
        if len(self.reward_history) > 1:
            reward_trend = self.reward_history[-1] - self.reward_history[-2]
        else:
            reward_trend = 0.0
        
        bitrate_pressure = min(1.0, self.current_frame / max(1, self.num_frames)) # Encoding progress
        encoding_difficulty = np.mean([abs(r) for r in self.reward_history[-5:]]) if self.reward_history else 0.0
        
        global_features.extend([reward_trend, bitrate_pressure, encoding_difficulty])
        
        # Ensure we have exactly 16 global features
        while len(global_features) < self.global_features:
            global_features.append(0.0)
        global_features = global_features[:self.global_features]
        
        return global_features

    def _action_to_qp_offsets(self, action: np.ndarray) -> np.ndarray:
        """Convert discrete action to QP offsets"""
        # Map from [0, QP_MAX-QP_MIN] to [QP_MIN, QP_MAX]
        qp_offsets = action + QP_MIN
        qp_offsets = np.clip(qp_offsets, QP_MIN, QP_MAX)
        
        # Validate QP offsets
        validate_array(qp_offsets, f"QP offsets for frame {self.current_frame}")
        
        return qp_offsets

    def _calculate_reward(
        self, feedback: Dict, action_request: Dict, qp_offsets: np.ndarray
    ) -> float:
        """Calculate reward based on encoding feedback"""
        try:
            # Get encoded frame data
            encoded_frame_data = feedback["encoded_frame_data"]
            
            # Get YCbCr PSNR for the current frame
            y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(
                self.current_frame, encoded_frame_data, self.baseline_heighest_psnr
            )
            
            # Validate PSNR values
            for psnr_val, psnr_name in [(y_psnr, "Y"), (cb_psnr, "Cb"), (cr_psnr, "Cr")]:
                if math.isnan(psnr_val) or math.isinf(psnr_val):
                    raise InvalidRewardError(
                        f"Invalid {psnr_name} PSNR ({psnr_val}) for frame {self.current_frame}"
                    )
            
            # Get byte usage difference
            byte_saved, current_usage = self.av1_runner.get_byte_usage_diff(
                action_request["picture_number"]
            )
            
            # Validate byte usage values
            if math.isnan(byte_saved) or math.isinf(byte_saved):
                raise InvalidRewardError(f"Invalid byte_saved ({byte_saved}) for frame {self.current_frame}")
            if math.isnan(current_usage) or math.isinf(current_usage) or current_usage <= 0:
                raise InvalidRewardError(f"Invalid current_usage ({current_usage}) for frame {self.current_frame}")
            
            # Calculate PSNR improvements
            y_psnr_improvement = y_psnr - self.y_psnr_list[self.current_frame]
            cb_psnr_improvement = cb_psnr - self.cb_psnr_list[self.current_frame]
            cr_psnr_improvement = cr_psnr - self.cr_psnr_list[self.current_frame]
            
            # Validate improvements
            for improvement, name in [
                (y_psnr_improvement, "Y PSNR improvement"),
                (cb_psnr_improvement, "Cb PSNR improvement"), 
                (cr_psnr_improvement, "Cr PSNR improvement")
            ]:
                if math.isnan(improvement) or math.isinf(improvement):
                    raise InvalidRewardError(f"Invalid {name} ({improvement}) for frame {self.current_frame}")
            
            # Calculate reward components
            a = 1
            b = c = 0.5
            d = 2
            
            byte_efficiency = byte_saved / current_usage
            
            # Validate byte efficiency
            if math.isnan(byte_efficiency) or math.isinf(byte_efficiency):
                raise InvalidRewardError(
                    f"Invalid byte efficiency ({byte_efficiency}) for frame {self.current_frame}. "
                    f"byte_saved: {byte_saved}, current_usage: {current_usage}"
                )
            
            # Calculate final reward
            reward = (
                y_psnr_improvement * a
                + cb_psnr_improvement * b
                + cr_psnr_improvement * c
                + byte_efficiency * d
            )
            
            # Final reward validation
            reward_details = {
                "y_psnr": y_psnr,
                "cb_psnr": cb_psnr,
                "cr_psnr": cr_psnr,
                "y_psnr_improvement": y_psnr_improvement,
                "cb_psnr_improvement": cb_psnr_improvement,
                "cr_psnr_improvement": cr_psnr_improvement,
                "byte_saved": byte_saved,
                "current_usage": current_usage,
                "byte_efficiency": byte_efficiency
            }
            
            validate_reward(reward, self.current_frame, reward_details)
            
            return reward
            
        except Exception as e:
            raise InvalidRewardError(f"Failed to calculate reward for frame {self.current_frame}: {e}")

    def _start_encoder_thread(self):
        """Start encoder in separate thread"""
        if self.encoder_thread and self.encoder_thread.is_alive():
            # print("Waiting for previous encoder thread to terminate...")
            self.encoder_thread.join(timeout=20.0)

        self.encoder_thread = threading.Thread(
            target=self._run_encoder, daemon=True, name="EncoderThread"
        )
        self.encoder_thread.start()
        # Give the encoder thread a brief head start before returning to main thread
        threading.Event().wait(0.05)  # Wait 50ms (adjust as needed)

    def _run_encoder(self):
        """Run encoder in separate thread"""
        print("Starting encoder thread...")
        self.av1_runner.run_SVT_AV1_encoder()
        print(f"Encoder thread completed (thread id: {threading.get_ident()})")

    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Maximum frames reached
        if self.current_frame >= self.num_frames:
            self.terminated = True
