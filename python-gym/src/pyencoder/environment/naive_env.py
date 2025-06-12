import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2, av
import gymnasium as gym
import numpy as np
from pyencoder.environment.av1_running_env import Av1RunningEnv
from pyencoder.utils import video_reader
from pyencoder.utils.video_reader import VideoReader
from sympy import false

# Constants
QP_MIN, QP_MAX = -3, 3  # delta QP range which will be action
SB_SIZE = 64  # superblock size


# @dataclass()
# class Av1GymInfo:
#     actions: np.ndarray
#     reward: float


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
        self.av1_running_env = Av1RunningEnv(video_path=video_path)
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

        # Observation space = previous frame summary
        # Observation space: 4 features per superblock (from get_frame_state)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4, self.num_superblocks),
            # 4 features:
            # 0: Y-component variance of all superblocks in the frame
            # 1: Horizontal motion vector of all superblocks in the frame
            # 2: Vertical motion vector of all superblocks in the frame
            # 3: Beta (example metric) of all superblocks in the frame
            dtype=np.float32,
        )

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
        self.av1_running_env.run_SVT_AV1_encoder(output_path=f"{str(output_dir)}/baseline_output.ivf")
        self.y_psnr_list = []
        self.cb_psnr_list = []
        self.cr_psnr_list = []
        
        self.save_baseline_frame_psnr(
            baseline_video_path=f"{str(output_dir)}/baseline_output.ivf"
        )
        
    def save_baseline_frame_psnr(self, baseline_video_path: str | Path):
        """Calculate and save PSNR for baseline frames"""
        baseline_video_path = str(baseline_video_path)
        container = av.open(baseline_video_path)
        stream = container.streams.video[0]

        total_frames = int(stream.frames)
        assert total_frames == self.num_frames, (
            f"Baseline video frame count {total_frames} does not match expected {self.num_frames}"
        )

        assert self.y_psnr_list == [], "PSNR lists should be empty before saving baseline PSNRs"
        assert self.cb_psnr_list == [], "PSNR lists should be empty before saving baseline PSNRs"
        assert self.cr_psnr_list == [], "PSNR lists should be empty before saving baseline PSNRs"

        for frame_number, frame in enumerate(container.decode(stream)):
            # Convert frame to YCbCr and get numpy arrays for Y, Cb, Cr
            ycbcr = frame.to_ndarray(format="rgb24")
            ycbcr = cv2.cvtColor(ycbcr, cv2.COLOR_RGB2YCrCb)

            # Get YCbCr PSNR for the current frame
            y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(frame_number, ycbcr)

            # Append to lists
            self.y_psnr_list.append(y_psnr)
            self.cb_psnr_list.append(cb_psnr)
            self.cr_psnr_list.append(cr_psnr)
        
        assert len(self.y_psnr_list) == self.num_frames, (
            f"Expected {self.num_frames} Y PSNR values, got {len(self.y_psnr_list)}"
        )
        assert len(self.cb_psnr_list) == self.num_frames, (
            f"Expected {self.num_frames} Cb PSNR values, got {len(self.cb_psnr_list)}"
        )
        assert len(self.cr_psnr_list) == self.num_frames, (
            f"Expected {self.num_frames} Cr PSNR values, got {len(self.cr_psnr_list)}"
        )
        print("Baseline frame PSNRs saved:")
        
    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        print("Resetting environment...")

        # Reset episode state
        self.current_frame = 0
        self.current_episode_reward = 0.0
        self.terminated = False
        self.frame_history.clear()

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

        # Convert action to QP offsets
        qp_offsets = self._action_to_qp_offsets(action)

        # Wait for encoder to request action for this frame
        action_request = self.av1_running_env.wait_for_action_request(
            timeout=self.queue_timeout
        )

        if action_request is None:
            print("No action request received - episode terminated")
            self.terminated = True
            return self._get_current_observation(), 0.0, True, False, {'timeout': True}
        
        frame_number = action_request['picture_number']

        # Send action response to encoder
        self.av1_running_env.send_action_response(qp_offsets.tolist())

        # Wait for encoding feedback
        feedback = self.av1_running_env.wait_for_feedback(timeout=self.queue_timeout)

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

        # Calculate reward
        reward = self._calculate_reward(feedback, action_request, qp_offsets)
        self.current_episode_reward += reward

        # Update episode state
        self.current_frame += 1

        # Check termination conditions
        self._check_termination_conditions()

        # Get next observation
        next_obs = self._get_current_observation()

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

    def _get_initial_observation(self) -> np.ndarray:
        """Get initial observation for episode start"""
        # Get first frame state
        frame_state = self.video_reader.get_x_frame_state(frame_number=0)
        return np.array(frame_state, dtype=np.float32)

    def _get_current_observation(self) -> np.ndarray:
        """Get current observation based on current frame"""
        frame_state = self.video_reader.get_x_frame_state(
            frame_number=self.current_frame
        )
        return np.array(frame_state, dtype=np.float32)

    def _action_to_qp_offsets(self, action: np.ndarray) -> np.ndarray:
        """Convert discrete action to QP offsets"""
        # Map from [0, QP_MAX-QP_MIN] to [QP_MIN, QP_MAX]
        qp_offsets = action + QP_MIN
        return np.clip(qp_offsets, QP_MIN, QP_MAX)

    def _calculate_reward(
        self, feedback: Dict, action_request: Dict, qp_offsets: np.ndarray
    ) -> float:
        """Calculate reward based on encoding feedback"""

        # get rgb com
        encoded_frame_data = feedback["encoded_frame_data"]

        y_psnr, cb_psnr, cr_psnr = self.video_reader.ycrcb_psnr(
            self.current_frame, encoded_frame_data
        )
        
        # TODO: a reward function that balances quality and bitrate
        a = 1
        b = c = 0.5
        d = 2
        byte_saved, current_usage = self.av1_running_env.get_byte_usage_diff(
            action_request["picture_number"]
        )
        
        y_psnr_improvement = y_psnr - self.y_psnr_list[self.current_frame]
        cb_psnr_improvement = cb_psnr - self.cb_psnr_list[self.current_frame]
        cr_psnr_improvement = cr_psnr - self.cr_psnr_list[self.current_frame]

        return y_psnr_improvement * a + \
                cb_psnr_improvement * b + \
                cr_psnr_improvement * c + \
                byte_saved / current_usage * d

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
        self.av1_running_env.run_SVT_AV1_encoder()
        print(f"Encoder thread completed (thread id: {threading.get_ident()})")

    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Maximum frames reached
        if self.current_frame >= self.num_frames:
            self.terminated = True
