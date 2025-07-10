import io
from queue import Queue
from pathlib import Path
from enum import Enum
from ..pyencoder import SuperBlockInfo
import threading
from dataclasses import dataclass

import av
import cv2
import numpy as np
import av1gym.pyencoder as encoder

class FrameType(Enum):
    KEY_FRAME        = 0
    INTER_FRAME      = 1
    INTRA_ONLY_FRAME = 2
    S_FRAME          = 3

@dataclass
class Observation:
    superblocks: list[SuperBlockInfo]
    frame_type: FrameType
    picture_number: int
    frames_to_key: int
    frames_since_key: int
    buffer_level: int

@dataclass
class Action:
    skip: bool
    offsets: list[int] | None

@dataclass
class Feedback:
    """
    encoded_frame_data: np.ndarray, shape (3/2 * H, W)
    """
    picture_number: int
    bitstream_size: int
    encoded_frame_data: np.ndarray

global the_only_object
the_only_object = None

def picture_feedback_trampoline(bitstream: bytes, size: int, picture_number: int):
    assert the_only_object is not None
    the_only_object.picture_feedback(bitstream, size, picture_number)

def get_deltaq_offset_trampoline(
        sbs: list[SuperBlockInfo], 
        frame_type: int, 
        picture_number: int, 
        frames_to_key: int,
        frames_since_key: int,
        buffer_level: int
) -> list[int]:
    assert the_only_object is not None
    return the_only_object.get_deltaq_offset(sbs, frame_type, picture_number, frames_to_key, frames_since_key, buffer_level)

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
                "Av1Runner instance already exists. Only one instance is allowed."
            )
        the_only_object = self

        self.video_path = video_path
        self.bytes_keeper = {}
        self.previous_training_bytes_keeper = {}  # Store previous training bytes
        self.all_bitstreams = io.BytesIO()  # Holds joined bitstream data

        w, h = frame_dims_from_file(video_path)
        self.sb_total_count = superblocks_from_dims(w, h)

        self.first_round = True  # Flag for the first round of encoding
        self.first_round_byte_usage = {}  # Store byte usage for the first round

        # Synchronization
        self.observation_queue: Queue[Observation] = Queue(maxsize=1) # Encoder provides observation
        self.action_queue: Queue[Action] = Queue(maxsize=1) # RL provides action
        self.feedback_queue: Queue = Queue() # Encoder provides feedback to RL
        self.encoder_thread: threading.Thread | None = None # Encoding thread

    def run(self, output_path: str | None = None, block: bool = False):
        """
        Start the encoder in a new thread.
        If block is True, wait for the encoder to finish.
        """
        if self.encoder_thread is not None and self.encoder_thread.is_alive():
            print("Waiting for previous encoder thread to terminate...")
            self.encoder_thread.join(timeout=20.0)

        self.encoder_thread = threading.Thread(
            target=self._run_encoder,
            args=(output_path,),
            daemon=True,
            name="EncoderThread"
        )
        self.encoder_thread.start()

        if block:
            self.encoder_thread.join()

    def _run_encoder(self, output_path: str | None = None):
        print("Starting encoder thread...")
        self.reset()
        self.register_callbacks()

        args = {
            "input": self.video_path,
            "pred_struct": 1,
            "rc": 2,
            "tbr": 100,
            "enable_stat_report": True,
        }

        if output_path:
            args["b"] = output_path

        encoder.run(**args)

    def join(self):
        if not self.encoder_thread or not self.encoder_thread.is_alive():
            return
        else:
            self.encoder_thread.join()

    def register_callbacks(self):
        encoder.register_callbacks(
            get_deltaq_offset=get_deltaq_offset_trampoline,
            picture_feedback=picture_feedback_trampoline,
        )

    def reset(self):
        """
        Reset the callback state, clearing the bytes_keeper and all_bitstreams.
        This is typically called at the start of a new encoding session.
        """
        self.previous_training_bytes_keeper = self.bytes_keeper.copy()
        self.bytes_keeper.clear()
        self.all_bitstreams.close()
        self.all_bitstreams = io.BytesIO()

        # Clear queues
        while not self.observation_queue.empty():
            self.observation_queue.get_nowait()

        while not self.action_queue.empty():
            self.action_queue.get_nowait()

        while not self.feedback_queue.empty():
            self.feedback_queue.get_nowait()

    def join_bitstreams(self):
        joined_bitstream_num = 0
        while joined_bitstream_num in self.bytes_keeper.keys():
            self.all_bitstreams += self.bytes_keeper[joined_bitstream_num]
            joined_bitstream_num += 1

    def picture_feedback(self, bitstream: bytes, size: int, picture_number: int):
        """
        Callback for receiving encoded picture data from the C encoder.
        This sends feedback to the RL environment.
        """
        self.bytes_keeper[picture_number] = bitstream
        encoded_frame_data = self.get_last_frame(bitstream=bitstream)

        # Prepare feedback for RL environment
        feedback_data = Feedback(
            picture_number = picture_number,
            bitstream_size = size,
            encoded_frame_data = encoded_frame_data,
        )

        # Send feedback to RL environment (non-blocking)
        self.feedback_queue.put_nowait(feedback_data)

    def get_last_frame(self, bitstream):
        byte_file = self.all_bitstreams
        byte_file.write(bitstream)
        byte_file.seek(0)
        container = av.open(byte_file, 'r')
        last_frame = None
        for frame in container.decode(video=0):
            last_frame = frame

        assert last_frame != None
        ycrcb_array = last_frame.to_ndarray(format="yuv420p") # (3/2 * H, W)

        # if the last frame is a keyframe, we can write the bitstream to the file
        if last_frame.key_frame:
            byte_file.close()
            self.all_bitstreams = io.BytesIO()  # get a new bytefile
            self.all_bitstreams.write(bitstream)  # write the keyframe to bytefile
        container.close()
        return ycrcb_array

    def get_deltaq_offset(
        self, 
        sbs: list[SuperBlockInfo], 
        frame_type: int, 
        picture_number: int,
        frames_to_key: int,
        frames_since_key: int,
        buffer_level: int
    ) -> list[int]:
        """
        Callback to get QP offsets for superblocks in a frame.
        This method MUST return immediately as the encoder waits synchronously.
        """
        # Request action from RL environment
        observation = Observation(
            picture_number=picture_number,
            superblocks=sbs,
            frame_type=FrameType(frame_type),
            frames_to_key=frames_to_key,
            frames_since_key=frames_since_key,
            buffer_level=buffer_level
        )

        # Send action request to RL environment (blocking call)
        self.observation_queue.put_nowait(observation)

        # Wait for RL response
        action = self.action_queue.get()

        # Dummy offsets to skip
        if action.skip:
            return [114514] * self.sb_total_count

        if action.offsets is None:
            raise ValueError(f"Action response is null")

        if len(action.offsets) != self.sb_total_count:
            raise ValueError(f"Action response length mismatch. Expected {self.sb_total_count}, got {len(action.offsets)}")

        return action.offsets

    def wait_for_next_observation(self) -> Observation:
        """
        Wait for action request from encoder.
        Called by RL environment to get the next frame to process.
        """
        return self.observation_queue.get()

    def send_action_response(self, *, action: list[int] | None = None, skip = False):
        """
        Send action response to encoder.
        Called by RL environment to provide QP offsets.
        """
        self.action_queue.put(Action(
            skip=skip,
            offsets=action
        ))

    def wait_for_feedback(self) -> Feedback:
        """
        Wait for feedback from encoder.
        Called by RL environment to get encoding results.
        """
        return self.feedback_queue.get()

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
        
    def save_bitstream_to_file(self, output_path: str, interrupt: bool = False):
        """
        Save the bitstream to a file.
        If not interrupted, it will save current bitstream data to ivf file.
        If interrupted, it will save the previous training bytes to ivf file.
        
        If the desierd bitestream is not complete, nothing will be saved and a warning will be given
        Args:
            output_path (str): The path to save the bitstream file. Must end with .ivf.
            interrupt (bool): If True, save the previous training bytes instead of current.
        Raises:
            ValueError: If the output path does not end with .ivf.
        """
        if not output_path.endswith(".ivf"):
            raise ValueError("Output path must end with .ivf")

        # Save previous training bytes
        # To make the bitstream playable, prepend the IVF header and frame headers
        frames = list(self.previous_training_bytes_keeper.values())
        if not frames:
            bitstream_data = b""
        else:
            # Try to get width/height from the first frame using PyAV
            container = av.open(io.BytesIO(frames[0]))
            stream = next(iter(container.streams.video))
            width = stream.width
            height = stream.height
            container.close()
            bitstream_data = ivf_header(len(frames), width, height)
            for i, frame in enumerate(frames):
                bitstream_data += ivf_frame_header(frame, i)
                bitstream_data += frame

        if not bitstream_data:
            print("No bitstream data to save.")
            return

        with open(output_path, "wb") as f:
            f.write(bitstream_data)
        print(f"Bitstream saved to {output_path}")

def ivf_header(num_frames: int, width: int, height: int, fourcc: bytes = b'AV01'):
    header = b'DKIF'  # signature
    header += (0).to_bytes(2, 'little')  # version
    header += (32).to_bytes(2, 'little')  # header size
    header += fourcc  # fourcc
    header += width.to_bytes(2, 'little')
    header += height.to_bytes(2, 'little')
    header += (30).to_bytes(4, 'little')  # timebase denominator
    header += (1).to_bytes(4, 'little')   # timebase numerator
    header += num_frames.to_bytes(4, 'little')
    header += (0).to_bytes(4, 'little')  # unused
    return header

def ivf_frame_header(frame_bytes: bytes, pts: int):
    return len(frame_bytes).to_bytes(4, 'little') + pts.to_bytes(8, 'little')

def superblocks_from_dims(width: int, height: int, block_size: int = 64) -> int:
    """
    Return the number of super-blocks (block_size × block_size tiles)
    that cover a frame of size (width × height).

    Uses ceiling division so partially-covered edges count as full blocks.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    blocks_w = (width  + block_size - 1) // block_size
    blocks_h = (height + block_size - 1) // block_size
    return blocks_w * blocks_h

def frame_dims_from_capture(cap: cv2.VideoCapture) -> tuple[int, int]:
    """
    Grab the first frame from an *open* cv2.VideoCapture and
    return (width, height).  Restores the original frame pointer.
    """
    if not cap.isOpened():
        raise ValueError("cv2.VideoCapture is not opened")

    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ok, frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos) # restore

    if not ok:
        raise ValueError("Could not read a frame from capture")

    height, width = frame.shape[:2]
    return width, height

def frame_dims_from_file(path: str | Path) -> tuple[int, int]:
    """
    Open the video at *path*, read its first frame, and return (width, height).
    Closes the capture automatically.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")
        return frame_dims_from_capture(cap)
    finally:
        cap.release()