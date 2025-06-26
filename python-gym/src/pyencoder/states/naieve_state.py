from .__templete import State_templete
from numpy import ndarray
from typing import Any, Dict, List, Optional

class NaiveState(State_templete):
    """
    A naive implementation of the State_templete class.
    This class provides a simple way to handle states without complex processing.
    """

    def __init__(self, frame: Optional[ndarray] = None, width: Optional[int] = None, height: Optional[int] = None,
                 num_sb: Optional[int] = None, SB_SIZE: int = 64, **kwargs: Any):
        """
        Initialize the NaiveState with flexible arguments.
        Only one of frame, (width and height), or num_sb should be provided.
        """
        self.SB_SIZE = SB_SIZE
        if frame is not None:
            h, w = frame.shape[:2]
        elif width is not None and height is not None:
            h, w = height, width
        elif num_sb is not None:
            self.num_sb = num_sb
            return
        else:
            raise ValueError("One of frame, (width and height), or num_sb must be provided.")

        self.num_sb = ((h + self.SB_SIZE - 1) // self.SB_SIZE) * ((w + self.SB_SIZE - 1) // self.SB_SIZE)

    @classmethod
    def from_frame(cls, frame: ndarray, SB_SIZE: int = 64, **kwargs: Any):
        return cls(frame=frame, SB_SIZE=SB_SIZE, **kwargs)

    @classmethod
    def from_shape(cls, width: int, height: int, SB_SIZE: int = 64, **kwargs: Any):
        return cls(width=width, height=height, SB_SIZE=SB_SIZE, **kwargs)

    @classmethod
    def from_num_sb(cls, num_sb: int, SB_SIZE: int = 64, **kwargs: Any):
        return cls(num_sb=num_sb, SB_SIZE=SB_SIZE, **kwargs)

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