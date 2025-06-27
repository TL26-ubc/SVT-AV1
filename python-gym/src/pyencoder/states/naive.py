from .__templete import State_templete
from numpy import ndarray
from typing import Any, Dict, List, Optional
import cv2
import gymnasium as gym

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

    @staticmethod
    def get_observation(frame: Optional[ndarray], 
                        SB_SIZE: int = 64,
                        **kwargs) -> ndarray:
        """
        Get the current observation of the state.
        
        Parameters:
            frame (Optional[ndarray]): The current frame or observation.
            SB_SIZE (int): Size of the state buffer, default is 64.
            **kwargs: Additional keyword arguments for processing the frame.
        
        Returns:
            ndarray: A 1D numpy array with any size, handling inf or nan values if present.
        """
        if frame is None:
            return ndarray(SB_SIZE)  # Return an empty array if no frame is provided
        return frame.flatten()[:SB_SIZE]  # Flatten and limit to SB_SIZE elements

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
            dtype='float32'
        )