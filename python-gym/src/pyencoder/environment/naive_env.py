import queue
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import cv2

from pyencoder.environment.utils import _probe_resolution
from pyencoder.utils.video_reader import VideoReader

from pyencoder.utils.sb_processing import *

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
        *,
        lambda_rd: float = 0.1,
        av1_runner: callable = None,
    ):
        super().__init__()
        self.video_path = Path(video_path)
        self.cv2_video_cap = cv2.VideoCapture(str(self.video_path))

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

        self._frame_report_q: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)
        self._episode_done = threading.Event()
        self._encoder_thread: threading.Thread | None = None
        self._frame_action: np.ndarray | None = None
        self._next_frame_idx = 0
        self._terminated = False

        self.av1_runner = av1_runner
        if self.av1_runner is None:
            raise ValueError("av1_runner function must be provided.")
        self.av1_runner()

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[dict, dict]:
        super().reset(seed=seed)
        self.close()
        self._terminated = False
        self._next_frame_idx = 0
        self._episode_done.clear()

        # Spawn encoder worker
        self._encoder_thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._encoder_thread.start()

        # Return first observation which is the first frame
        obs = {self._next_frame_idx: get_x_frame_state(
            self._next_frame_idx, self.cv2_video_cap, block_size=SB_SIZE
        )}
        return obs, {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("Call reset() before step() after episode ends.")

        if action.shape != (4, self.num_superblocks):
            raise ValueError(
                f"Action shape {action.shape} does not match expected shape "
                f"(4, {self.num_superblocks})."
            )
        # wait for a action to complete 
        # wake up the encoder thread for a frame to complete QP mapping 
        
        # TODO: wake up when next request is recieved or when terminated
        
        # wait for a feedback 
        
        # send action to encoder
        # self._action_q.put(action.astype(np.int32, copy=False))
        obs, reward, self._terminated, _, info = self.get_frame_feedback(
            self._frame_report_q.get()
        )
        self.send_action(action)

        return obs, reward, self._terminated, False, info

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.close
    def close(self):
        if self._encoder_thread and self._encoder_thread.is_alive():
            self._episode_done.set()
            self._encoder_thread.join(timeout=1.0)

        # drain queues
        # for q in (self._action_q, self._frame_report_q):
        #     while not q.empty():
        #         q.get_nowait()

        self._encoder_thread = None

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
    def send_action(self, action: np.ndarray):
        return action, action.size

    def get_frame_feedback(self, frame_report: Dict[str, Any]):
        # Wait for encoder to finish the frame
        report = self._frame_report_q.get()  # dict with stats + next obs
        reward = self._reward_fn(report)  # scalar
        obs = report["next_obs"]

        self._terminated = report["is_last_frame"]
        self._next_frame_idx += 1

        info: dict = {}

        return obs, reward, self._terminated, False, info

    # Reward function
    def _reward_fn(self, rpt: Dict[str, Any]) -> float:
        return -float(rpt["bits"]) + self.lambda_rd * float(rpt["psnr"])

import torch

class DQNAgent:
    def __init__(self, num_superblocks, action_dim, state_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.num_superblocks = num_superblocks
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = self._build_model().to(self.device)
        self.target_net = self._build_model().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = []
        self.batch_size = 32
        self.max_memory = 10000
        self.update_target_steps = 100
        self.step_count = 0

    def _build_model(self):
        # Simple MLP for demonstration
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.state_dim * self.num_superblocks, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.action_dim * self.num_superblocks)
        )

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            # Random action in the correct shape
            return np.random.randint(QP_MIN, QP_MAX + 1, size=(self.num_superblocks,))
        state = torch.tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        q_values = q_values.view(self.num_superblocks, self.action_dim)
        actions = torch.argmax(q_values, dim=1).cpu().numpy() + QP_MIN
        return actions

    def store(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.memory[i] for i in batch))

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).view(self.batch_size, self.num_superblocks, self.action_dim)
        action_indices = (actions - QP_MIN).unsqueeze(-1)
        q_selected = torch.gather(q_values, 2, action_indices).squeeze(-1).mean(dim=1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).view(self.batch_size, self.num_superblocks, self.action_dim)
            next_q_max = next_q_values.max(dim=2)[0].mean(dim=1)
            target = rewards + self.gamma * next_q_max * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_selected, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay