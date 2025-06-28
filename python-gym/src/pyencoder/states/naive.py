from .__templete import State_templete
from numpy import ndarray
from typing import Any, Dict, List, Optional
import cv2
import gymnasium as gym
from pyencoder.utils.video_reader import VideoReader
import numpy as np

# WARNING!! DO NOT CHANGE THE NAME OF THIS CLASS
class State(State_templete):
    """
    A naive implementation of the State_templete class.
    This class provides a simple way to handle states without complex processing.
    """

    def __init__(self, source_video_path: str, SB_SIZE: int = 64,
                 **kwargs: Any):
        """
        Initialize the NaiveState with flexible arguments.
        Only one of frame, (width and height), or num_sb should be provided.
        """
        self.SB_SIZE = SB_SIZE
        self.source_video_path = source_video_path
        video = cv2.VideoCapture(source_video_path)
        if not video.isOpened():
            raise ValueError(f"Cannot open video file: {source_video_path}")
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video.release()
        self.num_sb = ((h + self.SB_SIZE - 1) // self.SB_SIZE) * ((w + self.SB_SIZE - 1) // self.SB_SIZE)

    def initialize(self, 
                      video_reader: VideoReader,
                      SB_SIZE: int = 64
                          ):
        """
        Initialize the state for operations needed.
        Get the maximum state values for normalization.
        Parameters:
            video_reader (VideoReader): The video reader instance to extract state information.
            SB_SIZE (int): Size of the state buffer, default is 64.
        """
        frame_count = video_reader.get_frame_count()
        array_length = self.get_observation_length()
        self.max_values = np.full(array_length, -np.inf, dtype=np.float32)
        for i in range(frame_count):
            frame = video_reader.read_frame(frame_number=i)
            if frame is None or len(frame) == 0:
                continue
            obs = self.get_observation(frame, SB_SIZE=SB_SIZE)
            self.max_values = np.maximum(self.max_values, obs)

    def get_observation(self,
                        frame: ndarray, 
                        SB_SIZE: int = 64,
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

        for y in range(0, h, SB_SIZE):
            for x in range(0, w, SB_SIZE):  # follow encoder order, x changes first
                y_end = min(y + SB_SIZE, h)
                x_end = min(x + SB_SIZE, w)
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