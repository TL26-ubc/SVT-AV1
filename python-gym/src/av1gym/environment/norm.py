import numpy as np
import gymnasium as gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import cast
from .environment import Av1GymEnv, ObservationDict

class ObsNormWrapper(gym.ObservationWrapper):
    """
    Normalise a Dict observation with keys:
        • "superblock": (H, W, SB_FEATURES)  — per-SB data
        • "frame":      (FRAME_FEATURES,)    — global data
    RunningMeanStd keeps mean, stddev per feature channel; the moments fed into it are
    the mean/var across all super-blocks in the frame.
    """
    def __init__(self, env: Av1GymEnv, clip: float = 10.0, epsilon: float = 1e-8, update: bool = True):
        super().__init__(env)
        self.clip = clip
        self.eps = epsilon
        self.update = update

        obs_space: ObservationDict = cast(ObservationDict, env.observation_space)
        sb_channels = obs_space["superblock"].shape[-1]
        frame_dim = obs_space["frame"].shape[0]

        # vectors (length = num feature channels)
        self.rms_sb = RunningMeanStd(shape=(sb_channels,))
        self.rms_frame = RunningMeanStd(shape=(frame_dim,))

    def observation(self, observation: ObservationDict) -> dict:
        sb = observation["superblock"].astype(np.float32) # (H, W, C)
        fr = observation["frame"].astype(np.float32) # (F,)

        # update running statistics
        if self.update:
            # flatten sb grid and compute moments sb stat for the frame
            sb_flat = sb.reshape(-1, sb.shape[-1]) # (N_sb, C)
            sb_mean = sb_flat.mean(axis=0) # (C,)
            sb_var = sb_flat.var(axis=0) # (C,)
            n_sb = sb_flat.shape[0]

            self.rms_sb.update_from_moments(sb_mean, sb_var, n_sb)
            self.rms_frame.update(fr[None, :])

        # norm + clip
        sb_norm = (sb - self.rms_sb.mean) / np.sqrt(self.rms_sb.var + self.eps)
        fr_norm = (fr - self.rms_frame.mean) / np.sqrt(self.rms_frame.var + self.eps)

        sb_norm = np.clip(sb_norm, -self.clip, self.clip)
        fr_norm = np.clip(fr_norm, -self.clip, self.clip)

        return {"superblock": sb_norm, "frame": fr_norm}

    def save(self, file_path: str):
        np.savez(
            file_path,
            sb_mean=self.rms_sb.mean,
            sb_var=self.rms_sb.var,
            fr_mean=self.rms_frame.mean,
            fr_var=self.rms_frame.var,
        )

    def load(self, file_path: str):
        d = np.load(file_path)
        self.rms_sb.mean[:] = d["sb_mean"]
        self.rms_sb.var[:] = d["sb_var"]
        self.rms_frame.mean[:] = d["fr_mean"]
        self.rms_frame.var[:] = d["fr_var"]
        self.update = False
