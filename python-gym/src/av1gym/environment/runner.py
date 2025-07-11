import io
from queue import Queue
from pathlib import Path
from enum import Enum
import threading
from dataclasses import dataclass
import av
import numpy as np

from .utils.av1_decoder import Av1OnTheFlyDecoder
from .utils.video_reader import VideoReader
from ..pyencoder import SuperBlockInfo
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
    encoded_frame_data: np.ndarray, shape (3/2 * H, W) in yuv420 format
    """
    picture_number: int
    bitstream_size: int
    encoded_frame_data: np.ndarray

global the_only_object
the_only_object = None

def picture_feedback_trampoline(
    packet: bytes, 
    size: int, 
    picture_number: int
):
    assert the_only_object is not None
    the_only_object.picture_feedback(packet, size, picture_number)

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

    def __init__(self, video: VideoReader):
        global the_only_object
        if the_only_object is not None:
            raise RuntimeError(
                "Av1Runner instance already exists. Only one instance is allowed."
            )
        the_only_object = self
        self.video = video

        self.bitstream: list[bytes] = []
        self.last_bitstream: list[bytes] = []  # Store previous training bytes
        self.all_bitstreams = io.BytesIO()  # Holds joined bitstream data
        self.decoder = Av1OnTheFlyDecoder()

        # Synchronization
        self.observation_queue: Queue[Observation] = Queue(maxsize=1) # Encoder provides observation
        self.action_queue: Queue[Action] = Queue(maxsize=1) # RL provides action
        self.feedback_queue: Queue[Feedback] = Queue() # Encoder provides feedback to RL
        self.encoder_thread: threading.Thread | None = None # Encoding thread

    def run(self, output_path: str | None = None, block: bool = False):
        """
        Start the encoder in a new thread.
        If block is True, wait for the encoder to finish.
        """
        self.reset()
        self.register_callbacks()

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
    
        args = {
            "input": self.video.path,
            "pred_struct": 1,
            "rc": 2,
            "tbr": 100,
            "enable_stat_report": True,
        }

        if output_path:
            args["b"] = output_path

        encoder.run(**args)

    def join(self):
        if self.encoder_thread and self.encoder_thread.is_alive():
            self.encoder_thread.join()

    def register_callbacks(self):
        encoder.register_callbacks(
            get_deltaq_offset=get_deltaq_offset_trampoline,
            picture_feedback=picture_feedback_trampoline,
        )

    def reset(self):
        if self.encoder_thread is not None and self.encoder_thread.is_alive():
            print("Waiting for previous encoder thread to terminate...")
            self.encoder_thread.join(timeout=20.0)

        self.last_bitstream = self.bitstream.copy()
        self.bitstream.clear()
        self.all_bitstreams.close()
        self.all_bitstreams = io.BytesIO()

        # Clear queues
        while not self.observation_queue.empty():
            self.observation_queue.join()

        while not self.action_queue.empty():
            self.action_queue.get_nowait()

        while not self.feedback_queue.empty():
            self.feedback_queue.get_nowait()

    def picture_feedback(self, packet: bytes, size: int, picture_number: int):
        """
        Callback for receiving encoded picture data from the C encoder.
        This sends feedback to the RL environment.
        """
        self.bitstream.append(packet)
        last_frame = self.decoder.append(packet)
        yuv_framearray = last_frame.to_ndarray(format="yuv420p")

        # Prepare feedback for RL environment
        feedback_data = Feedback(
            picture_number = picture_number,
            bitstream_size = size,
            encoded_frame_data = yuv_framearray,
        )

        # Send feedback to RL environment (non-blocking)
        self.feedback_queue.put_nowait(feedback_data)

    def get_deltaq_offset(
        self, 
        superblocks: list[SuperBlockInfo], 
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
            superblocks=superblocks,
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
            return [114514] * len(superblocks)

        if action.offsets is None:
            raise ValueError(f"Action response is null")

        if len(action.offsets) != len(superblocks):
            raise ValueError(f"Action response length mismatch. Expected {len(superblocks)}, got {len(action.offsets)}")

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
    
    def save_bitstream_to_file(self, output_path: str):
        self.decoder.save_video(output_path)
