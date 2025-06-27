from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from numpy import ndarray
import gymnasium as gym

class State_templete(ABC):
    """
    Abstract base class for states in the PyEncoder framework.
    """
    
    @abstractmethod
    def __init__(self, source_video_path: Optional[str] = None, SB_SIZE: int = 64,
                 **kwargs: Any):
        """
        Initialize the state with any necessary parameters.
        
        Parameters:
            **kwargs: Additional keyword arguments for state initialization.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_observation(frame: Optional[ndarray], 
                        SB_SIZE: int = 64,
                        **kwargs) -> ndarray:
        """
        Get the current observation of the state.
        Parameters:
            frame (Optional[ndarray]): The current frame or observation.
            SB_SIZE (int): Size of the state buffer, default is 64.
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