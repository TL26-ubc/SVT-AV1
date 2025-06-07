import argparse
import pandas as pd # Import pandas
import pyencoder
import numpy as np
import cv2, av, io
from threading import Lock

global frame_counter, bytes_keeper
frame_counter = {}
bytes_keeper = {}

global video_width, video_height
video_width = 0
video_height = 0

global all_bitstreams, joined_bitstream_num, all_bytes_lock
# Store all_bitstreams as a bytes IO
all_bitstreams = io.BytesIO() # This will hold the joined bitstream data
all_bytes_lock = Lock() # Lock to ensure thread safety when accessing all_bitstreams
joined_bitstream_num = 0 # This will count how many bitstreams were joined

global y_buffers, cb_buffers, cr_buffers
y_buffers = [] # Buffers for Y plane data
cb_buffers = [] # Buffers for Cb plane data
cr_buffers = [] # Buffers for Cr plane data

def join_bitstreams():
    global all_bitstreams, joined_bitstream_num, bytes_keeper, all_bytes_lock
    if not bytes_keeper:
        print("No bitstream data to join.")
        return
    
    # check the current joined_bitstream_num, see if in bytes_keeper
    all_bytes_lock.acquire()
    while joined_bitstream_num in bytes_keeper.keys():
        all_bitstreams.write(bytes_keeper[joined_bitstream_num])
        joined_bitstream_num += 1
            
    all_bytes_lock.release()

def get_video_config(video_path):
    global video_width, video_height
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        video_width = stream.width
        video_height = stream.height


def get_deltaq_offset(
    sb_info_list: list,  # List of dictionaries, each with SB info
    sb_total_count: int,  # Total number of SBs, should len(sb_info_list)
    picture_number: int,  # Current picture number
    frame_type: int,  # 0 for INTER, 1 for I_SLICE (example, actual meaning depends on C)
) -> list[int]:  # Returns an int (deltaq in C, 0 for success typically)
    """
    Calculates QP offsets for all superblocks in a frame.

    Args:
        sb_info_list: A list of dictionaries. Each dictionary is expected to contain:
            'sb_org_x': int, 'sb_org_y': int, 'sb_width': int, 'sb_height': int,
            'sb_qindex': int (QP for this specific SB), 'beta': float (TPL beta).
        offset_list_to_fill: A pre-sized list that this function must populate with
                             integer QP offsets for each superblock.
        sb_total_count: The total number of superblocks.
        picture_number: The current picture/frame number.
        frame_type: Integer representing the frame type (e.g., 1 for I_SLICE).

    Returns:
        An integer status code (typically 0 for success).
    """
    # print(f"Python: get_deltaq_offset called for Frame {picture_number}, Type: {frame_type}, Total SBs: {sb_total_count}")
    if len(sb_info_list) != sb_total_count:
        raise RuntimeError("Python Error: List lengths do not match sb_total_count!")

    offset_list_to_fill = [0]*sb_total_count

    return offset_list_to_fill # Indicate success    

def picture_feedback(
    bitstream: bytes,
    size: int,
    picture_number: int
):
    # Store cumulative size if called multiple times
    global bytes_keeper

    if picture_number not in bytes_keeper:
        bytes_keeper[picture_number] = b""  # Initialize as empty bytes
    bytes_keeper[picture_number] += bitstream  # Append new bitstream data
    print(
        f"Python: picture_feedback for Frame {picture_number}, Appended chunk size: {len(bitstream)}, Total size for frame: {len(bytes_keeper[picture_number])}"
    )

    global frame_counter
    if picture_number not in frame_counter:
        frame_counter[picture_number] = 0
    frame_counter[picture_number] += 1
    
    join_bitstreams()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)
    parser.add_argument(
        "--output_excel",
        help="Output Excel file name",
        default="encoder_callback_data.xlsx",
    )

    args = parser.parse_args()
    print(args)
    
    # get the video configuration
    get_video_config(args.file)
    
    # Clear data from previous runs if any
    frame_counter.clear()
    bytes_keeper.clear() # Ensure bytes_keeper is cleared as a dictionary

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback
    )
    pyencoder.run(
        input=args.file, pred_struct=1, rc=2, tbr=100, enable_stat_report=True, 
        b="Output/output.ivf",  # Example output file for bitstream
    )
