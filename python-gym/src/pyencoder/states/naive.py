from .abstract import AbstractState
from numpy import ndarray
from typing import Any, Dict, List, Optional
import cv2
import gymnasium as gym
from pyencoder.utils.video_reader import VideoReader
import numpy as np

# WARNING!! DO NOT CHANGE THE NAME OF THIS CLASS
class NaiveState(AbstractState):
    """
    A naive implementation of the State_templete class.
    This class provides a simple way to handle states without complex processing.
    """

    def __init__(self, video_reader: VideoReader, sb_size: int = 64,
                 **kwargs: Any):
        """
        Initialize the NaiveState with flexible arguments.
        Only one of frame, (width and height), or num_sb should be provided.
        """
        self.sb_size = sb_size
        self.num_sb = video_reader.get_num_superblock()
        frame_count = video_reader.get_frame_count()
        array_length = self.get_observation_length()
        self.max_values = np.full(array_length, -np.inf, dtype=np.float32)
        for i in range(frame_count):
            frame = video_reader.read_frame(frame_number=i)
            if frame is None or len(frame) == 0:
                continue
            obs = self.get_observation(frame, SB_SIZE=sb_size)
            self.max_values = np.maximum(self.max_values, obs)

    def get_observation(self,
                        frame: ndarray,
                        **kwargs) -> ndarray:
        """
        Get the current observation of the state. Promise to normalize the observation.
        
        Parameters:
            frame ndarray: The current frame.
            SB_SIZE (int): Size of the state buffer, default is 64.
            **kwargs: Additional keyword arguments for processing the frame.
        
        Returns:
            ndarray: A 1D numpy array with any size, handling inf or nan values if present.
        """
        h, w = frame.shape[:2]
        y_comp_list = []
        h_mv_list = []
        v_mv_list = []
        beta_list = []

        for y in range(0, h, self.sb_size):
            for x in range(0, w, self.sb_size):  # follow encoder order, x changes first
                y_end = min(y + self.sb_size, h)
                x_end = min(x + self.sb_size, w)
                sb = frame[y:y_end, x:x_end]
                if sb.size == 0:
                    continue

                sb_y_var = np.var(sb[:, :, 0])  # Y-component variance
                sb_x_mv = np.mean(sb[:, :, 1])  # Horizontal motion vector
                sb_y_mv = np.mean(sb[:, :, 2])  # Vertical motion vector
                beta = np.mean(np.abs(sb))  # Example metric

                y_comp_list.append(sb_y_var)
                h_mv_list.append(sb_x_mv)
                v_mv_list.append(sb_y_mv)
                beta_list.append(beta)

        obs = np.array([y_comp_list, h_mv_list, v_mv_list, beta_list], dtype=np.float32).flatten()
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