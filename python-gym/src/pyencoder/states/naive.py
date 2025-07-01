from pyencoder import SuperBlockInfo
from .abstract import AbstractState
from typing import Any, Dict, List, Optional
from pyencoder.environment.av1_runner import Observation
import gymnasium as gym
from pyencoder.utils.video_reader import VideoReader
import numpy as np

class NaiveState(AbstractState):
    def __init__(
        self, 
        video_reader: VideoReader, 
        baseline_observations: list[Observation], 
        sb_size: int = 64,
        **kwargs: Any
    ):
        self.sb_size = sb_size
        self.num_sb = video_reader.get_num_superblock()
        self.frame_count = video_reader.get_frame_count()
        array_length = self.get_observation_length()
        self.max_values = np.full(array_length, -np.inf, dtype=np.float32)
        for raw_obs in baseline_observations:
            frame = video_reader.read_frame(frame_number=raw_obs.picture_number)
            if frame is None or len(frame) == 0:
                continue
            obs = self.get_observation(frame, raw_obs.superblocks, raw_obs.frame_type, raw_obs.picture_number)
            self.max_values = np.maximum(self.max_values, obs)

    def get_observation(
        self,
        frame: np.ndarray, # The current frame in yuv420p format, shape (3/2 * H, W).
        sbs: list[SuperBlockInfo], 
        frame_type: int,
        picture_number: int,
        **kwargs
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        assert h % 3 == 0, "Height must be a multiple of 3 for yuv420p format"
        h, w = h // 3 * 2, w  # Adjust for yuv420p format
        y_comp_list = []
        h_mv_list = []
        v_mv_list = []
        qindex_list = []

        sb_idx = 0
        for y in range(0, h, self.sb_size):
            for x in range(0, w, self.sb_size):  # follow encoder order, x changes first
                y_end = min(y + self.sb_size, h)
                x_end = min(x + self.sb_size, w)
                sb_y_component = frame[y:y_end, x:x_end]
                # sb_cb_component = frame[h + y // 2:h + y_end // 2, x:x_end]
                # sb_cr_component = frame[h + h // 2 + y // 2:h + h // 2 + y_end // 2, x:x_end]
                if sb_y_component.size == 0:
                    continue
                
                # Y-component variance
                sb_y_var = np.var(sb_y_component)
                # sb_cb_var = np.var(sb_cb_component)
                # sb_cr_var = np.var(sb_cr_component)

                y_comp_list.append(sb_y_var)
                h_mv_list.append(sbs[sb_idx]['sb_x_mv'])
                v_mv_list.append(sbs[sb_idx]['sb_y_mv'])
                qindex_list.append(sbs[sb_idx]['sb_qindex'])
                sb_idx += 1

        obs = np.array([y_comp_list, h_mv_list, v_mv_list, qindex_list], dtype=np.float32).flatten()
        # check for inf or nan values and handle them
        # if illegal, replace with self.max_values at the corresponding index
        obs = np.where(np.isfinite(obs), obs, self.max_values)
        return obs

    def get_observation_length(self) -> int:
        """
        Get the shape of the observation.
        
        Return an integer as the length of the 1D numpy array.
        """
        return 4 * self.num_sb
    
    def get_observation_space(self) -> gym.spaces.Space:
        """
        Get the observation space of the state.
        
        Returns:
            gym.spaces.Space: A gymnasium Space object representing the observation space.
        """
        return gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.get_observation_length(),),
            dtype=np.float32
        )