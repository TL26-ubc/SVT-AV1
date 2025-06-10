import queue
import threading
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import cv2

import random


from pyencoder.utils.sb_processing import (
    get_frame_state, 
    get_x_frame_state, 
    get_num_superblock,
    get_frame_psnr
)

# Constants
QP_MIN, QP_MAX = -3, 3  # delta QP range which will be action
SB_SIZE = 64  # superblock size


# Extending gymnasium's Env class
# https://gymnasium.farama.org/api/env/#gymnasium.Env
class Av1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        video_path: str | Path,
        encoder_callback,
        *,
        lambda_rd: float = 0.1,
        max_frames_per_episode: int = 1000,
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.cv2_video_cap = cv2.VideoCapture(str(self.video_path))

        self.encoder_callback = encoder_callback
        self.max_frames_per_episode = max_frames_per_episode;

        self.encoder_callback.set_rl_environment(self)

        self.lambda_rd = lambda_rd
        self._episode_done = threading.Event()


        self.num_superblocks = get_num_superblock(
            self.cv2_video_cap, block_size=SB_SIZE
        )

        # Action space = QP offset grid
        self.action_space = gym.spaces.MultiDiscrete(
            # num \in [QP_MAX, QP_MIN], there are num_superblocks of them
            [QP_MAX - QP_MIN + 1] * self.num_superblocks
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
        # self._action_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1) no need action ti

        # Episode management
        self.current_frame = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.terminated = False
        self.truncated = False

        # Frame data storage
        self.frame_history = []
        self.previous_frame_quality = None
        
        # Synchronization
        self.processing_lock = threading.RLock()
        self.encoder_thread = None
        self.encoder_running = False
   

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
        self.truncated = False
        self.frame_history.clear()
        self.previous_frame_quality = None

        self.cv2_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get initial observation (frame 0 state)
        initial_obs = self._get_initial_observation()
        
        # Start encoder in separate thread
        self._start_encoder_thread()

        info = {
            'frame_number': self.current_frame,
            'num_superblocks': self.num_superblocks,
            'episode_start': True
        }
        
        return initial_obs, info

    # # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    # def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
    #     if self._terminated:
    #         raise RuntimeError("Call reset() before step() after episode ends.")

    #     if action.shape != (4, self.num_superblocks):
    #         raise ValueError(
    #             f"Action shape {action.shape} does not match expected shape "
    #             f"(4, {self.num_superblocks})."
    #         )
    #     # wait for a action to complete 
    #     # wake up the encoder thread for a frame to complete QP mapping 
        
    #     # TODO: wake up when next request is recieved or when terminated
        
    #     # wait for a feedback 
        
    #     # send action to encoder
    #     # self._action_q.put(action.astype(np.int32, copy=False))
    #     obs, reward, self._terminated, _, info = self.get_frame_feedback(
    #         self._frame_report_q.get()
    #     )
    #     self.send_action(action)

    #     return obs, reward, self._terminated, False, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        if self.terminated or self.truncated:
            raise RuntimeError("Episode has ended. Call reset() before step().")
        
        # Validate action
        if action.shape != (self.num_superblocks,):
            raise ValueError(
                f"Action shape {action.shape} != expected ({self.num_superblocks},)"
            )
        
        # Convert action to QP offsets
        qp_offsets = self._action_to_qp_offsets(action)
        
        # Wait for encoder to request action for this frame
        print(f"Waiting for action request from encoder...")
        action_request = self.encoder_callback.wait_for_action_request(timeout=5.0)
        
        if action_request is None:
            print("No action request received - episode terminated")
            self.terminated = True
            return self._get_current_observation(), 0.0, True, False, {'timeout': True}
        
        frame_number = action_request['picture_number']

        print(f"Processing frame {frame_number} with {len(qp_offsets)} QP offsets")
        
        # Send action response to encoder
        self.encoder_callback.send_action_response(qp_offsets.tolist())
        
        # Wait for encoding feedback
        print(f"Waiting for feedback from encoder...")
        feedback = self.encoder_callback.wait_for_feedback(timeout=5.0)
        
        if feedback is None:
            print("No feedback received - episode terminated")
            self.terminated = True
            return self._get_current_observation(), 0.0, True, False, {'no_feedback': True}
        
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
        self.frame_history.append({
            'frame_number': frame_number,
            'qp_offsets': qp_offsets,
            'reward': reward,
            'feedback': feedback
        })

        info = {
            'frame_number': frame_number,
            'reward_components': self._get_reward_components(feedback, qp_offsets),
            'bitstream_size': feedback.get('bitstream_size', 0),
            'episode_frames': self.current_frame
        }
        
        print(f"Step completed - Frame: {frame_number}, Reward: {reward:.4f}")
        
        return next_obs, reward, self.terminated, self.truncated, info


    # https://gymnasium.farama.org/api/env/#gymnasium.Env.close
    def close(self):

        print("Closing environment...")

        if self.encoder_thread and self.encoder_thread.is_alive():
            self._episode_done.set()
            self.encoder_thread.join(timeout=2.0)

        if self.cv2_video_cap.isOpened():
            self.cv2_video_cap.release()
        
        print("Environment closed")
        
        # drain queues
        # for q in (self._action_q, self._frame_report_q):
        #     while not q.empty():
        #         q.get_nowait()


    # https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    def render(self):
        pass

    # Encoding
    # def _encode_loop(self):
    #     from mycodec import encode

    #     encode(
    #         str(self.video_path),
    #         on_superblock=self._on_superblock,
    #         on_frame_done=self._on_frame_done,
    #     )

    # use this in c callback
    # def send_action(self, action: np.ndarray):
    #     return action, action.size

    # def get_frame_feedback(self, frame_report: Dict[str, Any]):
    #     # Wait for encoder to finish the frame
    #     report = self._frame_report_q.get()  # dict with stats + next obs
    #     reward = self._reward_fn(report)  # scalar
    #     obs = report["next_obs"]

    #     self._terminated = report["is_last_frame"]
    #     self._next_frame_idx += 1

    #     info: dict = {}

    #     return obs, reward, self._terminated, False, info

    # # Reward function
    # def _reward_fn(self, rpt: Dict[str, Any]) -> float:
    #     return -float(rpt["bits"]) + self.lambda_rd * float(rpt["psnr"])

    def _get_initial_observation(self) -> np.ndarray:
        """Get initial observation for episode start"""
        try:
            # Get first frame state
            frame_state = get_x_frame_state(
                0, self.cv2_video_cap, block_size=SB_SIZE
            )
            return np.array(frame_state, dtype=np.float32)
        except Exception as e:
            print(f"Error getting initial observation: {e}")
            # Return zero observation as fallback
            return np.zeros((4, self.num_superblocks), dtype=np.float32)

    def _get_current_observation(self) -> np.ndarray:
        """Get current observation based on current frame"""
        try:
            frame_state = get_x_frame_state(
                self.current_frame, self.cv2_video_cap, block_size=SB_SIZE
            )
            return np.array(frame_state, dtype=np.float32)
        except Exception as e:
            print(f"Error getting current observation: {e}")
            # Return previous observation or zeros
            return np.zeros((4, self.num_superblocks), dtype=np.float32)
        
    def _action_to_qp_offsets(self, action: np.ndarray) -> np.ndarray:
        """Convert discrete action to QP offsets"""
        # Map from [0, QP_MAX-QP_MIN] to [QP_MIN, QP_MAX]
        qp_offsets = action + QP_MIN
        return np.clip(qp_offsets, QP_MIN, QP_MAX)
    
    def _calculate_reward(self, feedback: Dict, action_request: Dict, qp_offsets: np.ndarray) -> float:
        """Calculate reward based on encoding feedback"""
        
        try:
            bitstream_size = feedback.get('bitstream_size', 0)
            
            # Estimate quality (placeholder - in real implementation, use PSNR/SSIM)
            estimated_quality = self._estimate_quality(feedback, action_request)
            
            # Rate-distortion reward
            # Negative bitrate (minimize bits) + positive quality (maximize quality)
            bitrate_penalty = bitstream_size / 10000.0  # Normalize bitrate
            quality_reward = estimated_quality
            
            reward = -bitrate_penalty + self.lambda_rd * quality_reward
            
            # Temporal consistency reward
            temporal_reward = self._calculate_temporal_consistency_reward(qp_offsets)
            
            total_reward = reward + 0.1 * temporal_reward

            return float(total_reward)
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0

                
    def _estimate_quality(self, feedback: Dict, action_request: Dict) -> float:
        """Estimate quality based on QP and frame characteristics"""
        # quality estimation placeholder
        # TODO: implement actual quality estimation logic
        
        return random.uniform(0.0, 1.0)  # Random quality for placeholder
        
    
    def _calculate_temporal_consistency_reward(self, qp_offsets: np.ndarray) -> float:
        """Calculate reward for temporal consistency"""
        if len(self.frame_history) == 0:
            return 0.0
        
        # Compare with previous frame's QP offsets
        prev_qp_offsets = self.frame_history[-1]['qp_offsets']
        
        # Reward smooth transitions
        qp_diff = np.abs(qp_offsets - prev_qp_offsets)
        avg_diff = np.mean(qp_diff)
        
        # Exponential reward for consistency
        consistency_reward = np.exp(-avg_diff / 2.0)
        
        return consistency_reward

    def _check_termination_conditions(self):
        """Check if episode should terminate"""
        # Maximum frames reached
        if self.current_frame >= self.max_frames_per_episode:
            self.terminated = True
            return
        
        # Check video end
        total_frames = int(self.cv2_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.current_frame >= total_frames - 1:
            self.terminated = True
            return

    def _get_reward_components(self, feedback: Dict, qp_offsets: np.ndarray) -> Dict:
        """Get detailed reward components for logging"""
        bitstream_size = feedback.get('bitstream_size', 0)
        
        return {
            'bitrate_penalty': bitstream_size / 10000.0,
            'quality_reward': self._estimate_quality(feedback, {}),
            'temporal_reward': self._calculate_temporal_consistency_reward(qp_offsets),
            'total_frames': self.current_frame
        }

    def _start_encoder_thread(self):
        """Start encoder in separate thread"""
        if self.encoder_thread and self.encoder_thread.is_alive():
            return
        
        self.encoder_running = True
        self.encoder_thread = threading.Thread(
            target=self._run_encoder,
            daemon=True,
            name="EncoderThread"
        )
        self.encoder_thread.start()

    def _run_encoder(self):
        """Run encoder in separate thread"""
        try:
            print("Starting encoder thread...")
            self.encoder_callback.run_rl_encoder()
            print("Encoder thread completed")
        except Exception as e:
            print(f"Encoder thread error: {e}")
        finally:
            self.encoder_running = False


    
