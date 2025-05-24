import queue
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from pyencoder.environment.utils import _probe_resolution

# Constants
QP_MIN, QP_MAX = 0, 63
SB_SIZE = 64


# Extending gymnasium's Env class
# https://gymnasium.farama.org/api/env/#gymnasium.Env
class Av1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, video_path: str | Path, *, lambda_rd: float = 0.1, av1_):
        super().__init__()
        self.video_path = Path(video_path)
        self.lambda_rd = float(lambda_rd)

        self.w_px, self.h_px = _probe_resolution(self.video_path)
        self.w_sb = (self.w_px + SB_SIZE - 1) // SB_SIZE
        self.h_sb = (self.h_px + SB_SIZE - 1) // SB_SIZE

        # Action space = QP offset grid
        self.action_space = gym.spaces.MultiDiscrete(
            np.full((self.h_sb, self.w_sb), QP_MAX - QP_MIN + 1, dtype=np.int64)
        )

        # Observation space = previous frame summary
        self.observation_space = gym.spaces.Dict(
            {
                "bits": gym.spaces.Box(0, np.finfo("float32").max, (1,), np.float32),
                "psnr": gym.spaces.Box(0, np.finfo("float32").max, (1,), np.float32),
                # "frame": gym.spaces.Discrete(1_000_000), guess no frame number for now
            }
        )

        # RL/encoder communication
        self._action_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._frame_report_q: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=1)
        self._episode_done = threading.Event()
        self._encoder_thread: threading.Thread | None = None
        self._frame_action: np.ndarray | None = None
        self._next_frame_idx = 0
        self._terminated = False

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

        # Return first observation
        obs = {
            "bits": np.array([0.0], dtype=np.float32),
            "psnr": np.array([0.0], dtype=np.float32),
            "frame": 0,
        }
        return obs, {}

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.step
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        if self._terminated:
            raise RuntimeError("Call reset() before step() after episode ends.")

        if action.shape != (self.h_sb, self.w_sb):
            raise ValueError(
                f"Action grid shape {action.shape} != ({self.h_sb},{self.w_sb})"
            )

        # Send grid to encoder (blocks until encoder requests it)
        self._action_q.put(action.astype(np.int32, copy=False))

        # Wait for encoder to finish the frame
        report = self._frame_report_q.get()  # dict with stats + next obs
        reward = self._reward_fn(report)  # scalar
        obs = report["next_obs"]

        self._terminated = report["is_last_frame"]
        self._next_frame_idx += 1

        info: dict = {}
        return obs, reward, self._terminated, False, info

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.close
    def close(self):
        if self._encoder_thread and self._encoder_thread.is_alive():
            self._episode_done.set()
            self._encoder_thread.join(timeout=1.0)

        # drain queues
        for q in (self._action_q, self._frame_report_q):
            while not q.empty():
                q.get_nowait()

        self._encoder_thread = None

    # https://gymnasium.farama.org/api/env/#gymnasium.Env.render
    def render(self):
        pass

    # Encoding
    def _encode_loop(self):
        from mycodec import encode

        encode(
            str(self.video_path),
            on_superblock=self._on_superblock,
            on_frame_done=self._on_frame_done,
        )

    def _on_superblock(self, sb_stats: Dict[str, Any], sb_index: int) -> int:
        if self._frame_action is None:
            # Wait until RL has produced a grid for *this* frame
            self._frame_action = self._action_q.get()

        y, x = divmod(sb_index, self.w_sb)
        qp_int = int(self._frame_action[y, x])
        return qp_int

    def _on_frame_done(self, frame_report: Dict[str, Any]):
        obs_next = {
            "bits": np.array([frame_report["bits"]], dtype=np.float32),
            "psnr": np.array([frame_report["psnr"]], dtype=np.float32),
            "frame": self._next_frame_idx + 1,
        }

        self._frame_report_q.put(
            {
                **frame_report,
                "next_obs": obs_next,
                "is_last_frame": bool(frame_report.get("last_frame", False)),
            }
        )
        self._frame_action = None

        if frame_report.get("last_frame", False):
            self._episode_done.set()

    # Reward function
    def _reward_fn(self, rpt: Dict[str, Any]) -> float:
        return -float(rpt["bits"]) + self.lambda_rd * float(rpt["psnr"])
