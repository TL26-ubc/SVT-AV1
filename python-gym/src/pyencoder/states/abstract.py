from abc import ABC, abstractmethod
from typing import Any
from pyencoder import SuperBlockInfo
from pyencoder.utils.video_reader import VideoReader
import gymnasium as gym
from pyencoder.environment.av1_runner import Observation
import numpy as np


class AbstractState(ABC):
    """
    Abstract base class for states in the PyEncoder framework.
    """
    
    @abstractmethod
    def __init__(self, video_reader: VideoReader, baseline_observations: list[Observation], sb_size: int = 64,
                 **kwargs: Any):
        """
        Initialize the state with any necessary parameters.
        
        Parameters:
            **kwargs: Additional keyword arguments for state initialization.
        """
        pass

    @abstractmethod
    def get_observation(
        self, 
        frame: np.ndarray,
        superblocks: list[SuperBlockInfo], 
        frame_type: int,
        picture_number: int,
        **kwargs
    ) -> np.ndarray:
        """
        Get the current observation of the state.
        Parameters:
            frame ndarray: The current frame in yuv420p format, shape (3/2 * H, W).
                [0:H] is Y component, [H:H + H//2] is Cb component, and [H + H//2:H + H] is Cr component.
            superblocks: A list of superblock info, including locations and motion vectors.
            frame_type: An int representing the type of frame.
            picture_number: The current frame's number in the sequence.
            **kwargs: Additional keyword arguments for processing the frame.
        
        Return a 1D numpy array with any size
        
        Note that need to handle inf or nan values if present.
        """
        pass
    
    @abstractmethod
    def get_observation_length(self) -> int:
        """
        Get the shape of the observation.
        
        Return an integer as the length of the 1D numpy array.
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> gym.spaces.Space:
        """
        Get the observation space of the state.
        
        Return a gymnasium Space object representing the observation space.
        """
        pass