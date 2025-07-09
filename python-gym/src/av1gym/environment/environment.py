import threading
from pathlib import Path
from typing import TypedDict
import gymnasium as gym
import numpy as np
from .runner import Av1Runner, Feedback
from .utils.video_reader import VideoReader
from .constants import QP_MIN, QP_MAX
from typing import Any
import math

SB_FEATURES = 4
FRAME_FEATURES = 1

class ObservationDict(TypedDict):
    # Shape: (sb_h, sb_w, SB_FEATURES)
    superblock: np.ndarray
    # Shape: (FRAME_FEATURES,)
    frame: np.ndarray

# Extending gymnasium's Env class
# https://gymnasium.farama.org/api/env/#gymnasium.Env
class Av1GymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        *,
        lambda_rd: float = 0.1,
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.av1_runner = Av1Runner(video_path)

        self.lambda_rd = lambda_rd
        self._episode_done = threading.Event()

        # Initialize the VideoReader
        self.video_reader = VideoReader(path=video_path)

        self.sb_w, self.sb_h = self.video_reader.get_superblock_dims()
        self.num_sb = self.sb_w * self.sb_h
        self.num_frames = self.video_reader.get_frame_count()

        # Action space, QP offset grid
        self.action_space = gym.spaces.MultiDiscrete(
            # num \in [QP_MAX, QP_MIN], there are num_superblocks of them
            nvec=np.full(self.num_sb, QP_MAX - QP_MIN + 1, dtype=np.int32)
        )

        # Observation space, global and per sb observations
        self.observation_space = gym.spaces.Dict({
            "superblock": gym.spaces.Box(-np.inf, np.inf, (self.sb_h, self.sb_w, SB_FEATURES,), np.float32),
            "frame": gym.spaces.Box(-np.inf, np.inf, (FRAME_FEATURES,), np.float32),
        })

        # Episode management
        self.current_frame = 0
        self.terminated = False
        
    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    def reset(
        self, 
        *, 
        seed: int | None = None, 
        options: dict | None = None
    ) -> tuple[ObservationDict, dict]:
        print("Resetting environment...")

        super().reset(seed=seed)

        # Reset episode state
        self.current_frame = 0
        self.current_episode_reward = 0.0
        self.terminated = False

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

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    def step(
        self,
        action: np.ndarray
    ) -> tuple[ObservationDict, float, bool, bool, dict]:
        if self.terminated:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        # Validate action
        if action.shape != (self.num_sb,):
            raise ValueError(
                f"Action shape {action.shape} != expected ({self.num_sb},)"
            )
          
        # Validate action values
        validate_array(action, f"Action for frame {self.current_frame}")
        
        # Convert action to QP offsets
        qp_offsets = self._action_to_qp_offsets(action)
        
        # Send action response to encoder
        self.av1_runner.send_action_response(action=qp_offsets.tolist())
        
        # Wait for encoding feedback
        feedback = self.av1_runner.wait_for_feedback()

        # Calculate reward
        reward = self._calculate_reward(feedback, qp_offsets)
        self.current_episode_reward += reward

        # Update episode state
        self.current_frame += 1

        # Check termination conditions
        self._check_termination_conditions()
        
        # Get next observation with validation
        if self.terminated:
            # Episode has ended, return dummy observation
            next_obs: Any = dict()
            print(f"Episode terminated at frame {self.current_frame-1} (total frames: {self.num_frames})")
        else:
            # Get next observation with validation
            next_obs = self._get_next_observation()
        
        info = {
            "frame_number": self.current_frame,
            "reward": reward,
            "bitstream_size": feedback.bitstream_size,
            "episode_frames": self.current_frame,
        }

        return next_obs, reward, self.terminated, False, info

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.close
    def close(self):
        print("Closing environment...")

        if self.av1_runner:
            self.av1_runner.join()
        
        print("Environment closed.")

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    def render(self):
        pass
    
    def save_bitstream_to_file(self, output_path: str, interrupt: bool = False):
        """Save the bitstream to a file"""
        self.av1_runner.save_bitstream_to_file(output_path, interrupt=interrupt)

    def _get_next_observation(self) -> ObservationDict:
        """Get current observation based on current frame"""
        try:
            observation = self.av1_runner.wait_for_next_observation()
            y_plane, _, _ = self.video_reader.read_frame(frame_number=observation.picture_number)

            # Assert frame numbers match
            assert (
                observation.picture_number == self.current_frame
            ), f"observation frame {observation.picture_number} != current_frame {self.current_frame}"

            # Build observations for each sb
            n_sb = len(observation.superblocks)
            superblock_obs = np.empty((n_sb, SB_FEATURES), dtype=np.float32)

            for i, sb in enumerate(observation.superblocks):
                x0, y0 = sb["sb_org_x"], sb["sb_org_y"]
                w, h = sb["sb_width"], sb["sb_height"]

                luma_var = y_plane[y0 : y0 + h, x0 : x0 + w].var()

                superblock_obs[i] = (
                    sb["sb_qindex"],
                    sb["sb_x_mv"],
                    sb["sb_y_mv"],
                    luma_var
                )

            # Reshape from 2d to 3d tensor
            superblock_obs = superblock_obs.reshape((self.sb_h, self.sb_w, SB_FEATURES))
            
            frame_obs = np.array([
                observation.frame_type
            ], dtype=np.float32)

            return ObservationDict(
                superblock=superblock_obs,
                frame=frame_obs,
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to get current observation for frame {self.current_frame}: {e}")

    def _action_to_qp_offsets(self, action: np.ndarray) -> np.ndarray:
        """Convert discrete action to QP offsets"""
        # Map from [0, QP_MAX-QP_MIN] to [QP_MIN, QP_MAX]
        qp_offsets = action + QP_MIN
        qp_offsets = np.clip(qp_offsets, QP_MIN, QP_MAX)
        
        # Validate QP offsets
        validate_array(qp_offsets, f"QP offsets for frame {self.current_frame}")
        
        return qp_offsets

    def _calculate_reward(
        self, feedback: Feedback, qp_offsets: np.ndarray
    ) -> float:
        """Calculate reward based on encoding feedback"""
        try:
            # Get postencoded frame data
            postencoded_frame = feedback.encoded_frame_data

            # Convert to yuv planes
            postencoded_y_plane, _, _ = self.video_reader.get_yuv_planes(postencoded_frame)
            original_y_plane, _, _ = self.video_reader.read_frame(frame_number=feedback.picture_number)
            
            # Calculate mse as reward
            mse = VideoReader.compute_mse(postencoded_y_plane, original_y_plane)
            
            return mse
        except Exception as e:
            raise RuntimeError(f"Failed to calculate reward for frame {self.current_frame}: {e}")

    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Maximum frames reached
        if self.current_frame >= self.num_frames:
            self.terminated = True


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
        raise RuntimeError(
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
        raise RuntimeError(
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
    details: dict
) -> None:
    """Validate reward value"""
    if reward is None:
        raise RuntimeError(f"Reward is None for frame {frame_number}")
    
    if not isinstance(reward, (int, float, np.number)):
        raise RuntimeError(
            f"Reward must be numeric, got {type(reward)} for frame {frame_number}"
        )
    
    if math.isnan(reward):
        raise RuntimeError(
            f"Reward is NaN for frame {frame_number}. Details: {details}"
        )
    
    if math.isinf(reward):
        raise RuntimeError(
            f"Reward is infinite ({reward}) for frame {frame_number}. Details: {details}"
        )
    
    # Check for extremely large rewards that might indicate calculation errors
    if abs(reward) > 1000:
        print(f"Warning: Very large reward ({reward:.2f}) for frame {frame_number}")
