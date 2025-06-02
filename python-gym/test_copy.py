import argparse
import io
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
    return offset_list_to_fill


byte_buffer = io.BytesIO() # holds bistream data for the current frame

# a callback that will get frame byte streams from encoder
# want to render byte streams to YCbCr buffers
# the byte stream is ivf video format
def picture_feedback(bitstream: bytes, size: int, picture_number: int):
    global byte_buffer
    print(f"Picture feedback for frame {picture_number} with size {size} bytes")
    byte_buffer.write(bitstream)
    byte_buffer.seek(0)  # Reset buffer position for reading

    container = av.open(byte_buffer, format="ivf")

    for frame in container.decode(video=0):
        # Get Y, Cb, Cr planes (YCbCr == YUV420P)
        y_plane = frame.planes[0].to_bytes()
        cb_plane = frame.planes[1].to_bytes()
        cr_plane = frame.planes[2].to_bytes()

        # You can now use these planes as needed
        print(f"Decoded frame {frame.index}, size: {frame.width}x{frame.height}")

    # Reset buffer if needed
    byte_buffer.seek(0)
    byte_buffer.truncate(0)

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

    # Get video configuration
    get_video_config(args.file)

    # Clear previous run data
    frame_counter.clear()
    bytes_keeper.clear()

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)

    # Calculate total bytes from picture feedback
    total_bytes_from_pic_fb = sum(len(data) for data in bytes_keeper.values())
    if not bytes_keeper:
        print("No bitstream data recorded from picture_feedback.")
    else:
        print(
            f"Total bytes from picture_feedback bitstreams: {total_bytes_from_pic_fb} of {len(bytes_keeper)} frames"
        )

    # Create video from YCbCr buffers
    output_path = "output_video.avi"
    print(f"Creating video at {output_path} with {len(y_buffers)} frames...")

    if not y_buffers or not cb_buffers or not cr_buffers:
        print("No YCbCr buffers available to create video.")
        exit(1)

    frames = []
    for i in range(len(y_buffers)):
        if (
            y_buffers[i] is not None
            and cb_buffers[i] is not None
            and cr_buffers[i] is not None
        ):
            y_plane = Image.fromarray(y_buffers[i], mode="L")
            cb_plane = Image.fromarray(cb_buffers[i], mode="L")
            cr_plane = Image.fromarray(cr_buffers[i], mode="L")
            ycbcr_img = Image.merge("YCbCr", (y_plane, cb_plane, cr_plane))
            frames.append(np.array(ycbcr_img.convert("RGB")))
        else:
            print(f"Skipping frame {i} due to missing Y/Cb/Cr data.")

    print(f"Total frames to write: {len(frames)}")
    imageio.mimsave(output_path, frames, fps=30, codec="libx264")
    print(
        f"Video created successfully at {output_path} with {len(frames)} frames using imageio."
    )
