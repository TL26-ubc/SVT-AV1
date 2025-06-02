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

def update_ycbcr_buffers():
    global y_buffers, cb_buffers, cr_buffers
    global all_bitstreams, joined_bitstream_num
    current_frame = len(y_buffers)
    if joined_bitstream_num > current_frame:
        # new bitstream data is available, extract YCbCr data
        
        # get current file position in all_bitstreams
        position = all_bitstreams.tell()
        # Reset the BytesIO stream to the beginning
        all_bitstreams.seek(0)
        container = av.open(all_bitstreams)
        
        # Extract YCbCr data for missing frames
        for idx, frame in enumerate(container.decode(video=0)):
            if idx == current_frame:
                # Extract Y, Cb, Cr planes
                y_plane = frame.to_image().convert("YCbCr").split()[0]
                cb_plane = frame.to_image().convert("YCbCr").split()[1]
                cr_plane = frame.to_image().convert("YCbCr").split()[2]
                
                # Convert to numpy arrays and store in buffers
                y_buffers.append(np.array(y_plane))
                cb_buffers.append(np.array(cb_plane))
                cr_buffers.append(np.array(cr_plane))
                
                current_frame += 1
                
                if joined_bitstream_num <= current_frame:
                    # If we have reached the current joined_bitstream_num, stop
                    break
        # Restore the position of the BytesIO stream
        all_bitstreams.seek(position)

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
            
    update_ycbcr_buffers()
    all_bytes_lock.release()

def get_video_config(video_path):
    global video_width, video_height
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        video_width = stream.width
        video_height = stream.height

def get_deltaq_offset(
    sb_info_list: list,       # List of dictionaries, each with SB info
    sb_total_count: int,      # Total number of SBs, should len(sb_info_list)
    picture_number: int,      # Current picture number
    frame_type: int           # 0 for INTER, 1 for I_SLICE (example, actual meaning depends on C)
) -> list[int]:                     # Returns an int (deltaq in C, 0 for success typically)
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
        print("Python Error: List lengths do not match sb_total_count!")
        # Optionally raise an error or handle appropriately
        return -1 # Indicate an error

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
        bytes_keeper[picture_number] = b'' # Initialize as empty bytes
    bytes_keeper[picture_number] += bitstream # Append new bitstream data
    print(f"Python: picture_feedback for Frame {picture_number}, Appended chunk size: {len(bitstream)}, Total size for frame: {len(bytes_keeper[picture_number])}")
    
    global frame_counter
    if picture_number not in frame_counter:
        frame_counter[picture_number] = 0
    frame_counter[picture_number] += 1
    
    join_bitstreams()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)
    parser.add_argument("--output_excel", help="Output Excel file name", default="encoder_callback_data.xlsx")

    args = parser.parse_args()
    print(args)
    
    # get the video configuration
    get_video_config(args.file)
    
    # Clear data from previous runs if any
    frame_counter.clear()
    bytes_keeper.clear() # Ensure bytes_keeper is cleared as a dictionary

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset,
        picture_feedback=picture_feedback
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)
            
    total_bytes_from_pic_fb = 0
    if bytes_keeper:
        for frame_num, bitstream_data in bytes_keeper.items():
            total_bytes_from_pic_fb += len(bitstream_data)
    else:
        print("No bitstream data recorded from picture_feedback.")
    # This print might be redundant if frame_feedback's bytes_used is comprehensive
    print(f"Total bytes from picture_feedback bitstreams: {total_bytes_from_pic_fb} of {len(bytes_keeper)} frames")
    
    
    from PIL import Image
    import imageio

    # from the y_buffers, cb_buffers, cr_buffers, we can create a video
    output_path = "/home/tom/tl26/SVT-AV1/output_video.avi"
    print(f"Creating video at {output_path} with {len(y_buffers)} frames...")

    if not y_buffers or not cb_buffers or not cr_buffers:
        print("No YCbCr buffers available to create video.")
        exit(1)

    # Alternative approach: use imageio to write video instead of cv2


    # Prepare frames as RGB numpy arrays
    frames = []
    for i in range(len(y_buffers)):
        if (
            y_buffers[i] is not None and
            cb_buffers[i] is not None and
            cr_buffers[i] is not None
        ):
            y_plane = Image.fromarray(y_buffers[i], mode='L')
            cb_plane = Image.fromarray(cb_buffers[i], mode='L')
            cr_plane = Image.fromarray(cr_buffers[i], mode='L')
            ycbcr_img = Image.merge('YCbCr', (y_plane, cb_plane, cr_plane))
            rgb_img = ycbcr_img.convert('RGB')
            frames.append(np.array(rgb_img))
        else:
            print(f"Skipping frame {i} due to missing Y/Cb/Cr data.")
    
    print(f"Total frames to write: {len(frames)}")
    # Write video using imageio
    imageio.mimsave(output_path, frames, fps=30, codec='libx264')
    print(f"Video created successfully at {output_path} with {len(frames)} frames using imageio.")
    
    