import cv2, pyencoder
import numpy as np
from typing import List, Dict, Any

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
    sb_info_list: list[dict],
    sb_total_count: int,
    picture_number: int,
    frame_type: int
) -> list[int]:
    """
    Callback to get QP offsets for superblocks in a frame.
    
    Args:
        sb_info_list (list[dict]): List of dictionaries containing superblock info.
        offset_list_to_fill (list[int]): List to fill with QP offsets.
        sb_total_count (int): Total number of superblocks.
        picture_number (int): Current picture/frame number.
        frame_type (int): Frame type (e.g., I_SLICE).
    
    Returns:
        list[int]: The filled offset list.
    """
    return the_only_object.get_deltaq_offset(
        sb_info_list, sb_total_count, picture_number, frame_type
    )

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
        self.bytes_keeper = {}
        self.all_bitstreams = bytearray()  # Store all bitstreams as a bytearray
        self.joined_bitstream_num = 0  # Counter for joined bitstreams
        
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

    def join_bitstreams(self):
        while joined_bitstream_num in self.bytes_keeper.keys():
            self.all_bitstreams += self.bytes_keeper[joined_bitstream_num]
            joined_bitstream_num += 1

    def picture_feedback(self,
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
        assert picture_number not in self.bytes_keeper, \
            f"Picture number {picture_number} already exists in bytes_keeper."
        self.bytes_keeper[picture_number] = bitstream  # Store the bitstream
        # print(f"Picture feedback received for frame {picture_number}, size: {size}")

    def get_deltaq_offset(
        self,
        sb_info_list: list[dict],
        sb_total_count: int,
        picture_number: int,
        frame_type: int
    ) -> list[int]:
        """
        Callback to get QP offsets for superblocks in a frame.
        
        Args:
            sb_info_list (list[dict]): List of dictionaries containing superblock info.
            offset_list_to_fill (list[int]): List to fill with QP offsets.
            sb_total_count (int): Total number of superblocks.
            picture_number (int): Current picture/frame number.
            frame_type (int): Frame type (e.g., I_SLICE).
        
        Returns:
            list[int]: The filled offset list.
        """
        if len(sb_info_list) != sb_total_count:
            raise RuntimeError("List lengths do not match sb_total_count!")
        # TODO: add support to notify C whether use original method or or model prediction

        # Fill the offset list with dummy values for now
        offset_list_to_fill = [114514] * sb_total_count
        return offset_list_to_fill