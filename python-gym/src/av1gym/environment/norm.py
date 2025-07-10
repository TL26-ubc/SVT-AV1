import numpy as np
import gymnasium as gym
from stable_baselines3.common.running_mean_std import RunningMeanStd
from typing import Generic, TypeVar, TypedDict
import torch as th
from enum import Enum

from .environment import Av1GymEnv, RawObservationDict
from .runner import FrameType

SB_FEATURES = 5
FRAME_FEATURES = 4

# Observation dict will be converted into th.Tensors by gymnasium
ArrayT = TypeVar(
    "ArrayT",
    np.ndarray,
    th.Tensor,
)

class ObservationDict(TypedDict, Generic[ArrayT]):
    superblock: ArrayT # continuous superblock level features (sb_h, sb_w, SB_FEATURES,)
    frame: ArrayT # continuous frame level features (FRAME_FEATURES,)
    frame_type: ArrayT # onehot array [0, 0, 0, 1]

class Av1GymObsNormWrapper(gym.ObservationWrapper):
    env: Av1GymEnv

    def __init__(self, env: Av1GymEnv, clip: float = 10.0, epsilon: float = 1e-8, update: bool = True):
        super().__init__(env)
        self.clip = clip
        self.eps = epsilon
        self.update = update
        
        self.frame_w = env.frame_w
        self.frame_h = env.frame_h
        self.sb_w = env.sb_w
        self.sb_h = env.sb_h

        self.observation_space = gym.spaces.Dict({
            "superblock": gym.spaces.Box(-np.inf, np.inf, (self.sb_h, self.sb_w, SB_FEATURES,), np.float32),
            "frame": gym.spaces.Box(-np.inf, np.inf, (FRAME_FEATURES,), np.float32),
            "frame_type": gym.spaces.MultiBinary(len(FrameType)),
        })

        # vectors (length = num feature channels)
        self.rms_sb = RunningMeanStd(shape=(SB_FEATURES,))
        self.rms_frame = RunningMeanStd(shape=(FRAME_FEATURES,))

    def observation(self, observation: RawObservationDict) -> ObservationDict:
        y_plane = observation["original_frame"]["y_plane"]

        # Build observations for each sb
        n_sb = len(observation["superblocks"])
        superblock_obs = np.empty((n_sb, SB_FEATURES), dtype=np.float32)

        for i, sb in enumerate(observation["superblocks"]):
            x0, y0 = sb["sb_org_x"], sb["sb_org_y"]
            w, h = sb["sb_width"], sb["sb_height"]

            luma_var = y_plane[y0 : y0 + h, x0 : x0 + w].var()

            superblock_obs[i] = (
                sb["sb_qindex"],
                sb["sb_x_mv"],
                sb["sb_y_mv"],
                luma_var,
                sb["sb_8x8_distortion"]
            )

        # Reshape from 2d to 3d tensor
        superblock_obs = superblock_obs.reshape((self.sb_h, self.sb_w, SB_FEATURES))
        
        # Build cont. frame level observations
        frame_obs = np.array([
            observation["frame_number"],
            observation["frames_to_key"],
            observation["frames_since_key"],
            observation["buffer_level"]
        ], dtype=np.float32)

        network_obs = ObservationDict(
            superblock=superblock_obs,
            frame=frame_obs,
            frame_type=self.enum_to_onehot(FrameType, observation["frame_type"], dtype=np.float32)
        )

        return self._normalize(network_obs)
    
    def _normalize(self, observation: ObservationDict) -> ObservationDict:
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

        return ObservationDict(
            superblock=sb_norm, 
            frame=fr_norm,
            # Dont normalize onehot values
            frame_type=observation["frame_type"]
        )
    
    @staticmethod
    def enum_to_onehot(enum: type[Enum], value: int, *, dtype=np.float32) -> np.ndarray:
        onehot = np.zeros(len(enum), dtype=dtype)
        onehot[value] = 1.0
        return onehot

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