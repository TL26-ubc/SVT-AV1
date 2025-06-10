import cv2
import pyencoder
import numpy as np
from typing import List, Dict, Any, Optional
import threading
import queue
import time
from pyencoder.utils.sb_processing import get_frame_state, get_num_superblock

global the_only_object
the_only_object = None

def picture_feedback_trampoline(
                bitstream: bytes,
                size: int,
                picture_number: int
                ):
    """
    Callback for receiving encoded picture data from the C encoder.
    
    Args:
        bitstream (bytes): The encoded bitstream for the picture.
        size (int): The size of the bitstream.
        picture_number (int): The frame number of the picture.
    """
    the_only_object.picture_feedback(bitstream, size, picture_number)

def get_deltaq_offset_trampoline(
    picture_number: int
) -> list[int]:
    """
    Callback to get QP offsets for superblocks in a frame.
    
    Args:
        picture_number (int): Current picture/frame number.
    
    Returns:
        list[int]: The filled offset list.
    """
    return the_only_object.get_deltaq_offset(picture_number)

class EncoderCallback:
    """
    A class to handle callbacks from Python to C for encoding video frames.
    This class is designed to be used with a C encoder that requires specific
    callbacks for frame processing.
    """

    def __init__(self, args):
        global the_only_object
        if the_only_object is not None:
            raise RuntimeError("EncoderCallback instance already exists. Only one instance is allowed.")
        the_only_object = self
        
        self.video_path = args.file
        self.superblock_count = get_num_superblock(self.video_path)
        
        self.bytes_keeper = {}
        self.all_bitstreams = bytearray()  # Store all bitstreams as a bytearray
        self.joined_bitstream_num = 0  # Counter for joined bitstreams
        
        self.frame_stats = {}

        # Frame tracking
        self.frame_counter = 0
        self.episode_frames = []

        # Synchronization
        self.action_request_queue = queue.Queue(maxsize=1)  # Encoder requests action
        self.action_response_queue = queue.Queue(maxsize=1)  # RL provides action
        self.feedback_queue = queue.Queue(maxsize=10)  # Encoder provides feedback to RL
        
        # Threading synchronization
        self.action_lock = threading.RLock()
        self.current_frame_data = {}
        
        # Baseline mode flag
        self.baseline_mode = True 
        self.baseline_complete = False
        
    def run_SVT_AV1_encoder(self, output_path: str = None, first_round: bool = False):
        self.reset_parameter()
        self.register_callback()
        if output_path is not None:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True,
                b=output_path
            )
        else:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True
            )
            
        if first_round:
            print("First round completed, should be the result of the original SVT-AV1 encoder.")
            # TODO: do some calculation and save the result
            
    def register_callback(self):
        pyencoder.register_callbacks(
            get_deltaq_offset=get_deltaq_offset_trampoline,
            picture_feedback=picture_feedback_trampoline
        )

    def reset_parameter(self):
        """
        Reset the callback state, clearing the bytes_keeper and all_bitstreams.
        This is typically called at the start of a new encoding session.
        """
        self.bytes_keeper.clear()
        self.all_bitstreams = bytearray()
        self.joined_bitstream_num = 0  # Reset the counter for joined bitstreams
        self.frame_stats.clear()
        self.frame_counter = 0
        self.episode_frames.clear()

        # Clear queues
        while not self.action_request_queue.empty():
            try:
                self.action_request_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.action_response_queue.empty():
            try:
                self.action_response_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.feedback_queue.empty():
            try:
                self.feedback_queue.get_nowait()
            except queue.Empty:
                break

    def join_bitstreams(self):
        while joined_bitstream_num in self.bytes_keeper.keys():
            self.all_bitstreams += self.bytes_keeper[joined_bitstream_num]
            joined_bitstream_num += 1

    # def picture_feedback(self,
    #                 bitstream: bytes,
    #                 size: int,
    #                 picture_number: int
    #                 ):
    #     """
    #     Callback for receiving encoded picture data from the C encoder.
        
    #     Args:
    #         bitstream (bytes): The encoded bitstream for the picture.
    #         size (int): The size of the bitstream.
    #         picture_number (int): The frame number of the picture.
    #     """
    #     assert picture_number not in self.bytes_keeper, \
    #         f"Picture number {picture_number} already exists in bytes_keeper."
    #     self.bytes_keeper[picture_number] = bitstream  # Store the bitstream
    #     # print(f"Picture feedback received for frame {picture_number}, size: {size}")

    def picture_feedback(self, bitstream: bytes, size: int, picture_number: int):
        """
        Callback for receiving encoded picture data from the C encoder.
        This sends feedback to the RL environment.
        """

        self.bytes_keeper[picture_number] = bitstream
        frame_data = self.current_frame_data.get(picture_number, {})
        
        # Prepare feedback for RL environment
        feedback_data = {
            'picture_number': picture_number,
            'bitstream_size': size,
            'frame_data': frame_data,
            'timestamp': time.time()
        }

        # Send feedback to RL environment (non-blocking)
        if not self.baseline_mode:
            try:
                self.feedback_queue.put_nowait(feedback_data)
            except queue.Full:
                print(f"Warning: Feedback queue full for frame {picture_number}")
        
        # print(f"Picture feedback sent for frame {picture_number}, size: {size}")

    def get_deltaq_offset(self, picture_number: int) -> list[int]:
        """
        Callback to get QP offsets for superblocks in a frame.
        This method MUST return immediately as the encoder waits synchronously.
        """
        
        self.current_frame_data[picture_number] = {
            'timestamp': time.time()
        }
        
        sb_total_count = self.superblock_count
        
        if self.baseline_mode:
            # Return special value to use encoder's default method
            return [114514] * sb_total_count
        
        if not self.rl_env:
            # Fallback to default if no RL environment
            return [0] * sb_total_count
        
        # Request action from RL environment
        action_request = {
            'picture_number': picture_number,
            'timestamp': time.time()
        }
        
        try:
            # Send action request to RL environment (blocking call)
            self.action_request_queue.put(action_request, timeout=0.1)
            
            # Wait for RL response 
            action_response = self.action_response_queue.get(timeout=1.0)
            
            if len(action_response) != sb_total_count:
                print(f"Warning: Action response length mismatch. Expected {sb_total_count}, got {len(action_response)}")
                return [0] * sb_total_count
                
            return action_response
            
        except queue.Full:
            print(f"Warning: Action request queue full for frame {picture_number}")
            return [0] * sb_total_count
        except queue.Empty:
            print(f"Warning: No action response received for frame {picture_number}")
            return [0] * sb_total_count
        except Exception as e:
            print(f"Error in get_deltaq_offset: {e}")
            return [0] * sb_total_count
    
    def wait_for_action_request(self, timeout=None) -> Optional[Dict]:
        """
        Wait for action request from encoder.
        Called by RL environment to get the next frame to process.
        """
        try:
            return self.action_request_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def send_action_response(self, action_list: List[int]):
        """
        Send action response to encoder.
        Called by RL environment to provide QP offsets.
        """
        try:
            self.action_response_queue.put(action_list, timeout=0.1)
        except queue.Full:
            print("Warning: Action response queue full")
    
    def wait_for_feedback(self, timeout=None) -> Optional[Dict]:
        """
        Wait for feedback from encoder.
        Called by RL environment to get encoding results.
        """
        try:
            return self.feedback_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    

    def set_rl_environment(self, rl_env):
        """Set the RL environment after initialization"""
        self.rl_env = rl_env
        self.baseline_mode = False

    def run_baseline_encoder(self, output_path: str = None):
        """Run encoder in baseline mode to get reference performance"""
        print("Running baseline encoder...")
        self.reset_parameter()
        self.baseline_mode = True
        self.register_callback()
        
        if output_path:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True,
                b=output_path
            )
        else:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True
            )
        
        self.baseline_complete = True
        print("Baseline encoding completed")
    
    def run_rl_encoder(self, output_path: str = None):
        """Run encoder with RL control"""
        if not self.rl_env:
            raise RuntimeError("RL environment not set. Call set_rl_environment() first.")
            
        print("Running RL-controlled encoder...")
        self.reset_parameter()
        self.baseline_mode = False
        self.register_callback()
        
        if output_path:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True,
                b=output_path
            )
        else:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True
            )


    def get_frame_state_from_video(self, frame_number: int) -> np.ndarray:
        """Get frame state using sb_processing helper functions"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")
        
        # Get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Cannot read frame {frame_number}")
        
        frame_state = get_frame_state(frame, block_size=64)
        return np.array(frame_state, dtype=np.float32)