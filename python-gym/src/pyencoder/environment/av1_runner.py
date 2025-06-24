import io
import queue
import time
from typing import Any, Dict, List, Optional

import av
import cv2
import pyencoder
import numpy as np

global the_only_object
the_only_object = None


def picture_feedback_trampoline(bitstream: bytes, size: int, picture_number: int):
    """
    Callback for receiving encoded picture data from the C encoder.

    Args:
        bitstream (bytes): The encoded bitstream for the picture.
        size (int): The size of the bitstream.
        picture_number (int): The frame number of the picture.
    """
    the_only_object.picture_feedback(bitstream, size, picture_number)


def get_deltaq_offset_trampoline(picture_number: int) -> list[int]:
    """
    Callback to get QP offsets for superblocks in a frame.

    Args:
        picture_number (int): Current picture/frame number.
    Returns:
        list[int]: The filled offset list.
    """
    return the_only_object.get_deltaq_offset(picture_number)

def get_num_superblock(
    frame_or_video: np.ndarray | cv2.VideoCapture | str,
    block_size: int = 64
) -> int:
    """
    Get the number of superblocks in a video frame or video.

    Args:
        frame_or_video (np.ndarray or cv2.VideoCapture): The video frame or video capture object.
        block_size (int): Size of the blocks to be processed. Should be 64 in SVT-AV1.

    Returns:
        int: Number of superblocks in the frame or in the first frame of the video.
    """
    if isinstance(frame_or_video, np.ndarray):
        h, w = frame_or_video.shape[:2]
    elif isinstance(frame_or_video, cv2.VideoCapture):
        pos = frame_or_video.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = frame_or_video.read()
        if not ret:
            raise ValueError("Could not read frame from video.")
        h, w = frame.shape[:2]
        frame_or_video.set(cv2.CAP_PROP_POS_FRAMES, pos)
    elif isinstance(frame_or_video, str):
        video_cv2 = cv2.VideoCapture(frame_or_video)
        if not video_cv2.isOpened():
            raise ValueError(f"Could not open video file: {frame_or_video}")
        pos = video_cv2.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = video_cv2.read()
        if not ret:
            raise ValueError("Could not read frame from video.")
        h, w = frame.shape[:2]
        video_cv2.set(cv2.CAP_PROP_POS_FRAMES, pos)
        video_cv2.release()
    else:
        raise TypeError("Input must be a numpy.ndarray or cv2.VideoCapture.")
    num_blocks_h = (h + block_size - 1) // block_size
    num_blocks_w = (w + block_size - 1) // block_size
    return num_blocks_h * num_blocks_w

class Av1Runner:
    """
    A class to handle callbacks from Python to C for encoding video frames.
    This class is designed to be used with a C encoder that requires specific
    callbacks for frame processing.
    """

    def __init__(self, video_path):
        global the_only_object
        if the_only_object is not None:
            raise RuntimeError(
                "EncoderCallback instance already exists. Only one instance is allowed."
            )
        the_only_object = self

        self.video_path = video_path
        self.bytes_keeper = {}
        self.all_bitstreams = io.BytesIO()  # Holds joined bitstream data
        self.sb_total_count = get_num_superblock(video_path)
        self.first_round = True  # Flag for the first round of encoding
        self.first_round_byte_usage = {}  # Store byte usage for the first round

        # Synchronization
        self.action_request_queue = queue.Queue(maxsize=1)  # Encoder requests action
        self.action_response_queue = queue.Queue(maxsize=1)  # RL provides action
        self.feedback_queue = queue.Queue(maxsize=10)  # Encoder provides feedback to RL

    def run_SVT_AV1_encoder(self, output_path: str = None):
        self.reset_parameter()
        self.register_callback()
        if output_path is not None:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True,
                b=output_path,
            )
        else:
            pyencoder.run(
                input=self.video_path,
                pred_struct=1,
                rc=2,
                tbr=100,
                enable_stat_report=True,
            )

        if self.first_round:
            self.first_round = False
            print(
                "First round completed, should be the result of the original SVT-AV1 encoder."
            )

            # Store the byte usage for the first round
            for picture_number, bitstream in self.bytes_keeper.items():
                self.first_round_byte_usage[picture_number] = len(bitstream)

    def register_callback(self):
        pyencoder.register_callbacks(
            get_deltaq_offset=get_deltaq_offset_trampoline,
            picture_feedback=picture_feedback_trampoline,
        )

    def reset_parameter(self):
        """
        Reset the callback state, clearing the bytes_keeper and all_bitstreams.
        This is typically called at the start of a new encoding session.
        """
        self.bytes_keeper.clear()
        self.all_bitstreams.close()
        self.all_bitstreams = io.BytesIO()

        # Clear queues
        while not self.action_request_queue.empty():
            self.action_request_queue.get_nowait()

        while not self.action_response_queue.empty():
            self.action_response_queue.get_nowait()

        while not self.feedback_queue.empty():
            self.feedback_queue.get_nowait()

    def join_bitstreams(self):
        while joined_bitstream_num in self.bytes_keeper.keys():
            self.all_bitstreams += self.bytes_keeper[joined_bitstream_num]
            joined_bitstream_num += 1

    def picture_feedback(self, bitstream: bytes, size: int, picture_number: int):
        """
        Callback for receiving encoded picture data from the C encoder.
        This sends feedback to the RL environment.
        """

        self.bytes_keeper[picture_number] = bitstream
        if self.first_round:
            # If it's the first round, we don't process the bitstream
            return

        encoded_frame_data = self.get_last_frame(bitstream=bitstream)

        # Prepare feedback for RL environment
        feedback_data = {
            "picture_number": picture_number,
            "bitstream_size": size,
            "encoded_frame_data": encoded_frame_data,
        }

        # Send feedback to RL environment (non-blocking)
        self.feedback_queue.put_nowait(feedback_data)

        # print(f"Picture feedback sent for frame {picture_number}, size: {size}")

    def get_last_frame(self, bitstream):
        byte_file = self.all_bitstreams
        byte_file.write(bitstream)
        byte_file.seek(0)
        container = av.open(byte_file)
        last_frame = None
        for frame in container.decode(video=0):
            last_frame = frame

        assert last_frame != None
        img_array = last_frame.to_ndarray(format="rgb24")
        ycrcb_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)

        # if the last frame is a keyframe, we can write the bitstream to the file
        if last_frame.key_frame:
            byte_file.close()
            self.all_bitstreams = io.BytesIO()  # get a new bytefile
            self.all_bitstreams.write(bitstream)  # write the keyframe to bytefile
        container.close()
        return ycrcb_array

    def get_deltaq_offset(self, picture_number: int) -> list[int]:
        """
        Callback to get QP offsets for superblocks in a frame.
        This method MUST return immediately as the encoder waits synchronously.
        """

        if self.first_round:
            return [114514] * self.sb_total_count  # Dummy offsets for first round

        # Request action from RL environment
        action_request = {"picture_number": picture_number, "timestamp": time.time()}

        # Send action request to RL environment (blocking call)
        self.action_request_queue.put_nowait(action_request)

        # Wait for RL response
        action_response = self.action_response_queue.get(timeout=1000)

        if len(action_response) != self.sb_total_count:
            raise ValueError(
                f"Action response length mismatch. Expected {self.sb_total_count}, got {len(action_response)}"
            )

        return action_response

    def wait_for_action_request(self, timeout=None) -> Optional[Dict]:
        """
        Wait for action request from encoder.
        Called by RL environment to get the next frame to process.
        """
        return self.action_request_queue.get(timeout=timeout)

    def send_action_response(self, action_list: List[int]):
        """
        Send action response to encoder.
        Called by RL environment to provide QP offsets.
        """
        self.action_response_queue.put(action_list, timeout=0.1)

    def wait_for_feedback(self, timeout=None) -> Optional[Dict]:
        """
        Wait for feedback from encoder.
        Called by RL environment to get encoding results.
        """
        return self.feedback_queue.get(timeout=timeout)

    def get_byte_usage_diff(self, picture_number: int) -> tuple[int, int]:
        """
        Get the byte usage difference for a specific frame compared to the first round.
        Returns the difference in bytes used for encoding the frame.
        Args:
            picture_number (int): The frame number to check.
        Returns:
            (int, int): A tuple containing the difference in bytes and the current size of the frame.
            positive difference means the frame is larger than in the first round,
            negative difference means the frame is smaller.
        """
        if (
            picture_number in self.first_round_byte_usage
            and picture_number in self.bytes_keeper
        ):
            first_round_size = self.first_round_byte_usage[picture_number]
            current_size = len(self.bytes_keeper.get(picture_number, b""))
            return (first_round_size - current_size, current_size)
        assert False, f"Frame {picture_number} not found in first round byte usage"
