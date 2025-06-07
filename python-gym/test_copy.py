import argparse
import io
import pdb
from threading import Lock

import av
import cv2
import numpy as np
import pandas as pd
import pyencoder

# Global variables
frame_counter = {}
bytes_keeper = {}
video_width = 0
video_height = 0

all_bitstreams = io.BytesIO()  # Holds joined bitstream data
all_bytes_lock = Lock()  # Lock for thread safety
joined_bitstream_num = 0  # Counts how many bitstreams were joined

y_buffers = []  # Y plane data buffers
cb_buffers = []  # Cb plane data buffers
cr_buffers = []  # Cr plane data buffers


def get_video_config(video_path):
    global video_width, video_height
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        video_width = stream.width
        video_height = stream.height


pc_c = set()


def get_deltaq_offset(
    sb_info_list: list, sb_total_count: int, picture_number: int, frame_type: int
) -> list[int]:
    """
    Calculates QP offsets for all superblocks in a frame.

    Args:
        sb_info_list (list): List of dictionaries with SB info.
        sb_total_count (int): Total number of superblocks.
        picture_number (int): Current picture/frame number.
        frame_type (int): Frame type (0 for INTER, 1 for I_SLICE).

    Returns:
        list[int]: QP offsets for each superblock.
    """
    if len(sb_info_list) != sb_total_count:
        print("Python Error: List lengths do not match sb_total_count!")
        return [-1]

    offset_list_to_fill = [0] * sb_total_count
    pc_c.add(picture_number)
    return offset_list_to_fill


byte_buffer = {}


# a callback that will get frame byte streams from encoder
# want to render byte streams to YCbCr buffers
# the byte stream is ivf video format
def picture_feedback(bitstream: bytes, size: int, picture_number: int):
    global byte_buffer
    byte_buffer[picture_number] = bitstream


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", help="Input video file", default="Data/akiyo_qcif.y4m"
    )
    parser.add_argument(
        "--output_excel",
        help="Output Excel file name",
        default="encoder_callback_data.xlsx",
    )
    args = parser.parse_args()
    print(args)

    # Get video configuration
    get_video_config(args.file)

    # Clear previous run data
    frame_counter.clear()
    bytes_keeper.clear()

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)

    byte_file = io.BytesIO()
    cur_pos = 0
    for picture_number, bitstream in byte_buffer.items():
        byte_file.write(bitstream)  # Write the bitstream
        byte_file.seek(0)  # Reset position to the start
        container = av.open(byte_file)

        # get last frame from the container
        last_frame = None
        for frame in container.decode(video=0):
            last_frame = frame

        # write last frame to the png file
        last_frame.to_image().save(f"frame_{picture_number}.png")

        # if the last frame is a keyframe, we can write the bitstream to the file
        if last_frame.key_frame:
            byte_file.close()
            byte_file = io.BytesIO()
            byte_file.write(bitstream)


# state machine
# if 0th frame x
