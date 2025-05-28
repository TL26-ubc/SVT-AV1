import argparse
import os
import pandas as pd # Import pandas
import json # For potentially serializing complex data

import pyencoder

global frame_counter, value_keeper, bytes_keeper, excel_data_rows # Add excel_data_rows
frame_counter = {}
value_keeper = {}
bytes_keeper = {}
excel_data_rows = [] # Initialize a list to store data for Excel


# Helper function to get or create a data dictionary for a frame
def _get_frame_data_dict(picture_number):
    global excel_data_rows
    # Check if data for this frame already exists
    for row in excel_data_rows:
        if row.get("Frame Number") == picture_number:
            return row
    # If not, create a new one and add it
    new_row = {"Frame Number": picture_number}
    excel_data_rows.append(new_row)
    return new_row

def get_deltaq_offset(
    sb_info_list: list,       # List of dictionaries, each with SB info
    offset_list_to_fill: list,# List to be populated with qp_offsets by this function
    sb_total_count: int,      # Total number of SBs, should len(sb_info_list)
    picture_number: int,      # Current picture number
    frame_type: int           # 0 for INTER, 1 for I_SLICE (example, actual meaning depends on C)
) -> int:                     # Returns an int (deltaq in C, 0 for success typically)
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
    frame_data = _get_frame_data_dict(picture_number)
    frame_data["Frame Type (DeltaQ)"] = "INTRA" if frame_type == 1 else "INTER"
    
    # Store a summary of sb_info_list. For example, count or first SB's info.
    # Avoid storing the whole list if it's very large.
    if sb_info_list:
        frame_data["Superblock Info Summary (DeltaQ)"] = f"{len(sb_info_list)} SBs received. First SB qindex: {sb_info_list[0].get('sb_qindex', 'N/A')}, beta: {sb_info_list[0].get('beta', 'N/A'):.2f}"
    else:
        frame_data["Superblock Info Summary (DeltaQ)"] = "No SB info received"

    print(f"Python: get_deltaq_offset called for Frame {picture_number}, Type: {frame_type}, Total SBs: {sb_total_count}")
    if len(sb_info_list) != sb_total_count or len(offset_list_to_fill) != sb_total_count:
        print("Python Error: List lengths do not match sb_total_count!")
        # Optionally raise an error or handle appropriately
        return -1 # Indicate an error

    is_intra_frame = (frame_type == 1) # Assuming 1 means I_SLICE

    calculated_offsets_summary = []

    for i in range(sb_total_count):
        current_sb_info = sb_info_list[i]

        # Extract available info from the current_sb_info dictionary
        sb_org_x = current_sb_info.get("sb_org_x", 0)
        sb_org_y = current_sb_info.get("sb_org_y", 0)
        sb_width = current_sb_info.get("sb_width", 0)
        sb_height = current_sb_info.get("sb_height", 0)
        sb_qindex = current_sb_info.get("sb_qindex", 50) # SB's current QP
        beta = current_sb_info.get("beta", 0.5)

        # --- Print received SuperBlockInfo for verification ---
        if i == 0: # Print only for the first SB to reduce log spam
            print(f"  SB Index {i}:")
            print(f"    Raw SB Info: {current_sb_info}")
            print(f"    Parsed - Position: ({sb_org_x},{sb_org_y}), Size: {sb_width}x{sb_height}")
            print(f"    Parsed - SB_QP: {sb_qindex}, Beta: {beta:.4f}")
        # The old version had frame-level qindex and encoder_bit_depth. These are not
        # directly passed in the new (OOuii)i signature via sb_info_list.
        # If needed, the C bridge or SuperBlockInfo struct would need to be extended.
        # print(f"    Frame_QIndex (MISSING): unknown, Encoder Bit Depth (MISSING): unknown")

        # --- Example QP offset calculation logic (adapted) ---
        qp_offset = 0

        if beta > 0.8:  # High confidence from TPL, likely complex
            qp_offset = -2
        elif beta < 0.2: # Low confidence, possibly smooth
            qp_offset = 2
        else:
            # Frame-level qindex is not available directly.
            # Using sb_qindex for this part of the logic.
            if sb_qindex > 45: # If SB's current QP is already high
                qp_offset = -1
            elif sb_qindex < 25: # If SB's current QP is low
                qp_offset = 1

        if is_intra_frame:
            qp_offset -= 1 # Often, intra blocks benefit from slightly lower QP

        # Example region-based adjustment (ensure you know video dimensions if using absolute coordinates)
        # This part would require video_width/height which are not passed.
        # video_width = 176 # Example for qCIF
        # video_height = 144 # Example for qCIF
        # center_x, center_y = sb_org_x + sb_width // 2, sb_org_y + sb_height // 2
        # if center_x < video_width / 3 and center_y < video_height / 3: # Top-left region
        #     qp_offset -= 1

        qp_offset = max(-5, min(5, qp_offset)) # Clamp offset to a reasonable range

        print(f"    Decision: QP offset = {qp_offset}")
        
        offset_list_to_fill[i] = qp_offset
        if i < 5: # Store first 5 calculated offsets as an example
            calculated_offsets_summary.append(qp_offset)

    frame_data["Calculated QP Offsets (First 5)"] = str(calculated_offsets_summary)
    print(f"Python: get_deltaq_offset completed for Frame {picture_number}.\n")
    return 0 # Indicate success


def frame_feedback(
    picture_number: int,
    bytes_used: int,
    width: int,
    height: int,
    buffer_y: list,
    buffer_cb: list,
    buffer_cr: list,
):
    frame_data = _get_frame_data_dict(picture_number)
    frame_data["Bytes Used (FrameFb)"] = bytes_used
    frame_data["Width (FrameFb)"] = width
    frame_data["Height (FrameFb)"] = height
    frame_data["YUV Buffers (FrameFb)"] = f"Received Y:{len(buffer_y)}x{len(buffer_y[0]) if buffer_y else 0}, Cb, Cr"

    print(f"Frame {picture_number}: width={width}, height={height}, bytes_used={bytes_used}")
    # update rl model
    
    global frame_counter
    if picture_number not in frame_counter:
        frame_counter[picture_number] = 0
    frame_counter[picture_number] += 1
    
    global value_keeper
    value_keeper[picture_number] = (buffer_y, buffer_cb, buffer_cr)
    
    

def picture_feedback(
    bitstream: bytes,
    # size: int,
    picture_number: int
):
    frame_data = _get_frame_data_dict(picture_number)
    # Store cumulative size if called multiple times
    global bytes_keeper
    if picture_number in bytes_keeper:
        frame_data["Bitstream Size (PicFb)"] = frame_data.get("Bitstream Size (PicFb)", 0) + len(bitstream)
    else:
        frame_data["Bitstream Size (PicFb)"] = len(bitstream)


    if picture_number not in bytes_keeper:
        bytes_keeper[picture_number] = b'' # Initialize as empty bytes
    bytes_keeper[picture_number] += bitstream # Append new bitstream data
    print(f"Python: picture_feedback for Frame {picture_number}, Appended chunk size: {len(bitstream)}, Total size for frame: {len(bytes_keeper[picture_number])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)
    parser.add_argument("--output_excel", help="Output Excel file name", default="encoder_callback_data.xlsx")

    args = parser.parse_args()
    print(args)
    
    # Clear data from previous runs if any
    excel_data_rows.clear()
    frame_counter.clear()
    value_keeper.clear()
    bytes_keeper.clear() # Ensure bytes_keeper is cleared as a dictionary

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset,
        frame_feedback=frame_feedback,
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
            
    # --- Save collected data to Excel ---
    if excel_data_rows:
        df = pd.DataFrame(excel_data_rows)
        df.sort_values(by="Frame Number", inplace=True)
        df.set_index("Frame Number", inplace=True)
        try:
            df.to_excel(args.output_excel)
            print(f"Callback data saved to {args.output_excel}")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            print("Saving to CSV as fallback: encoder_callback_data.csv")
            df.to_csv("encoder_callback_data.csv")
    else:
        print("No data collected from callbacks to save to Excel.")
    # --- End of Excel saving ---

    # --- Attempt to render video from collected bitstreams ---
    if bytes_keeper:
        print("\nAttempting to reconstruct video from collected bitstreams...")

        # --- Retrieve video dimensions for IVF header ---
        video_width = 0
        video_height = 0
        # Try to get dimensions from excel_data_rows, which should be populated by frame_feedback
        if excel_data_rows:
            for row_data in sorted(excel_data_rows, key=lambda x: x.get("Frame Number", float('inf'))):
                if "Width (FrameFb)" in row_data and "Height (FrameFb)" in row_data:
                    video_width = row_data["Width (FrameFb)"]
                    video_height = row_data["Height (FrameFb)"]
                    if video_width > 0 and video_height > 0: # Ensure valid dimensions
                        break
        
        if video_width == 0 or video_height == 0:
            # Fallback: Try to infer from value_keeper if YUV data is there and consistent
            # This is less ideal as frame_feedback is the direct source for encoded dimensions
            print("Warning: Could not get dimensions from frame_feedback via excel_data_rows. Trying to infer from YUV buffers.")
            if value_keeper:
                first_frame_num = sorted(value_keeper.keys())[0]
                buffer_y, _, _ = value_keeper[first_frame_num]
                if buffer_y and isinstance(buffer_y, list) and len(buffer_y) > 0 and isinstance(buffer_y[0], list):
                    video_height = len(buffer_y)
                    video_width = len(buffer_y[0])
                    print(f"Inferred dimensions from YUV buffer: {video_width}x{video_height}")

        if video_width == 0 or video_height == 0:
            print("Error: Video dimensions (width/height) could not be determined. Cannot create IVF file.")
            # Set a flag or handle this error to skip ffmpeg processing if dimensions are missing
            can_reconstruct_video = False
        else:
            can_reconstruct_video = True
        # --- End of dimension retrieval ---

        if can_reconstruct_video:
            ivf_file = "reconstructed_av1_stream.ivf" # Changed extension
            output_video_file_mkv = "reconstructed_video_from_callbacks.mkv"
            output_video_file_mp4 = "reconstructed_video_from_callbacks.mp4"

            try:
                with open(ivf_file, "wb") as f:
                    # Write IVF File Header (32 bytes)
                    f.write(b'DKIF')  # Signature
                    f.write((0).to_bytes(2, 'little'))  # Version
                    f.write((32).to_bytes(2, 'little'))  # Header length
                    f.write(b'AV01')  # Codec FourCC
                    f.write(video_width.to_bytes(2, 'little'))
                    f.write(video_height.to_bytes(2, 'little'))
                    
                    # Frame rate and time scale (e.g., 30 fps)
                    # Using a common timebase like 1/1000 for PTS if actual rate is unknown
                    # Or, assume a fixed rate like 30fps. Let's use 30/1 for simplicity.
                    frame_rate_num = 30 
                    frame_rate_den = 1
                    f.write(frame_rate_num.to_bytes(4, 'little')) 
                    f.write(frame_rate_den.to_bytes(4, 'little')) 
                    
                    # Placeholder for number of frames, will be updated later
                    num_frames_ivf_pos = f.tell()
                    f.write((0).to_bytes(4, 'little'))  # Number of frames (placeholder)
                    f.write((0).to_bytes(4, 'little'))  # Unused

                    print("IVF Frame Data Summary (first 5 valid frames or fewer):")
                    pts_counter = 0 
                    frames_written_to_ivf = 0
                    MIN_VALID_FRAME_SIZE = 150 # Stricter minimum valid frame size

                    all_frames_from_keeper_modified = []
                    skipped_due_to_size_count = 0
                    skipped_frame_8 = False

                    for frame_num in sorted(bytes_keeper.keys()):
                        bitstream_data = bytes_keeper[frame_num]
                        original_len = len(bitstream_data)

                        if frame_num == 8:
                            skipped_frame_8 = True
                            if original_len > 2:
                                bitstream_data = bitstream_data[:-2]
                                print(f"Using trimmed Frame 8: Original size {original_len}, Trimmed size {len(bitstream_data)}")
                            else:
                                print(f"Frame 8 is too short ({original_len} bytes) to trim, using as is (will likely be skipped by size filter).")
                            # Update length for size check after potential trimming
                            current_len = len(bitstream_data) 
                        else:
                            current_len = original_len

                        if current_len >= MIN_VALID_FRAME_SIZE:
                            all_frames_from_keeper_modified.append((frame_num, bitstream_data))
                        else:
                            if frame_num != 8 : # Avoid double-counting if Frame 8 was already too small before trimming
                                print(f"Skipping Frame {frame_num} (size {current_len}B) - less than {MIN_VALID_FRAME_SIZE}B.")
                                skipped_due_to_size_count +=1
                            elif frame_num == 8 and original_len < MIN_VALID_FRAME_SIZE and current_len < MIN_VALID_FRAME_SIZE:
                                # If original Frame 8 was already too small and remained so after trimming (or wasn't trimmed)
                                print(f"Skipping trimmed Frame 8 (size {current_len}B) - less than {MIN_VALID_FRAME_SIZE}B.")
                                skipped_due_to_size_count +=1

                    if skipped_frame_8 and not any(f[0]==8 for f in all_frames_from_keeper_modified) and len(bytes_keeper.get(8, b'')) >= MIN_VALID_FRAME_SIZE:
                         # This case means Frame 8 was initially large enough, got trimmed, and the trimmed version was also large enough but somehow wasn't added.
                         # This should ideally not happen with current logic but is a safeguard print.
                         # Or, it means Frame 8 was large, trimmed, and then the trimmed version became too small.
                         if len(bytes_keeper.get(8,b'')[:-2]) < MIN_VALID_FRAME_SIZE and len(bytes_keeper.get(8, b'')) >= MIN_VALID_FRAME_SIZE:
                             print(f"Note: Trimmed Frame 8 was skipped as its new size {len(bytes_keeper.get(8,b'')[:-2])}B was less than {MIN_VALID_FRAME_SIZE}B.")
                             skipped_due_to_size_count +=1 # Count it if it was skipped due to trimming making it too small

                    print(f"Total frames skipped due to size: {skipped_due_to_size_count}")
                    frames_to_write = all_frames_from_keeper_modified
                                        
                    if not frames_to_write or frames_to_write[0][0] != 0:
                        print("Error: Frame 0 is missing, not selected due to size, or not first. Cannot proceed.")
                        # Check if Frame 0 specifically was skipped due to size
                        if 0 not in [f[0] for f in frames_to_write] and 0 in bytes_keeper and len(bytes_keeper[0]) < MIN_VALID_FRAME_SIZE:
                            print(f"Reason: Frame 0 (size {len(bytes_keeper[0])}B) was smaller than MIN_VALID_FRAME_SIZE ({MIN_VALID_FRAME_SIZE}B).")
                        can_reconstruct_video = False
                    else:
                        print(f"Preparing to write {len(frames_to_write)} frames (Frame 8 trimmed, min size {MIN_VALID_FRAME_SIZE}B) to IVF. Frames up to: {frames_to_write[-1][0]}")
                        pts_counter = 0
                        frames_written_to_ivf = 0

                    for frame_num, bitstream_data in frames_to_write:
                        frame_size = len(bitstream_data)
                        print(f"  Writing Frame {frame_num} (PTS {pts_counter}): {frame_size} bytes")
                        
                        f.write(frame_size.to_bytes(4, 'little'))
                        f.write(pts_counter.to_bytes(8, 'little')) 
                        
                        f.write(bitstream_data)
                        pts_counter += 1 # pts_counter will be 0 for the first frame
                        frames_written_to_ivf += 1
                    
                    # Go back and write the actual number of frames written (0 or 1)
                    f.seek(num_frames_ivf_pos)
                    f.write(frames_written_to_ivf.to_bytes(4, 'little'))
                    f.flush() # Ensure all data is written before getting size or closing
                
                print(f"AV1 stream packaged into IVF file: {ivf_file} ({frames_written_to_ivf} frames written, {os.path.getsize(ivf_file)} bytes)")

                # --- Add diagnostic print for the first valid frame's bitstream data ---
                if frames_to_write: # Check if there was any valid frame to write
                    first_valid_frame_num, first_valid_bitstream = frames_to_write[0]
                    print(f"First valid frame ({first_valid_frame_num}) bitstream (first 16 bytes hex):")
                    print(' '.join(f'{b:02x}' for b in first_valid_bitstream[:16]))
                # --- End of diagnostic print ---

                import subprocess
                # Attempt 1: Try to copy the AV1 stream directly from IVF into an MKV container
                ffmpeg_cmd_mkv = [
                    "ffmpeg",
                    "-y",
                    "-i", ivf_file, # Input is now the IVF file
                    "-c", "copy",      
                    output_video_file_mkv
                ]
                current_output_video_file = output_video_file_mkv

                print(f"Running ffmpeg command (Attempt 1 - copy from IVF to MKV): {' '.join(ffmpeg_cmd_mkv)}")
                process_mkv = subprocess.run(ffmpeg_cmd_mkv, capture_output=True, text=True)
                
                if process_mkv.returncode == 0:
                    print(f"Video successfully reconstructed (copied from IVF) and saved as {current_output_video_file}")
                else:
                    print(f"Attempt 1 (copy from IVF to MKV) failed (return code: {process_mkv.returncode}). FFmpeg stderr:\n{process_mkv.stderr}")
                    print("Attempting to re-encode to MP4 from IVF...")
                    
                    # Attempt 2: Re-encode to MP4 from IVF
                    ffmpeg_cmd_mp4 = [
                        "ffmpeg",
                        "-y",
                        "-i", ivf_file, # Input is now the IVF file
                        "-c:v", "libx264", 
                        output_video_file_mp4 
                    ]
                    current_output_video_file = output_video_file_mp4 
                    print(f"Running ffmpeg command (Attempt 2 - re-encode from IVF to MP4 defaults): {' '.join(ffmpeg_cmd_mp4)}")
                    process_mp4 = subprocess.run(ffmpeg_cmd_mp4, capture_output=True, text=True)

                    if process_mp4.returncode == 0:
                        print(f"Video successfully reconstructed (re-encoded from IVF) and saved as {current_output_video_file}")
                    else:
                        print(f"Attempt 2 (re-encode from IVF to MP4) also failed (return code: {process_mp4.returncode}):")
                        print("FFmpeg stdout:")
                        print(process_mp4.stdout)
                        print("FFmpeg stderr:")
                        print(process_mp4.stderr)
                        print(f"Please check if '{ivf_file}' is a valid IVF file and the AV1 data within is correct.")

            except FileNotFoundError:
                 print(f"Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
            except Exception as e:
                print(f"An error occurred during IVF creation or video reconstruction: {e}")
        elif not bytes_keeper: # This case was already there, kept for completeness
             print("No bitstream data was collected in bytes_keeper to reconstruct video.")
        # else: # This implies can_reconstruct_video is False due to missing dimensions
             # print("Video reconstruction skipped due to missing dimensions.") # Message already printed above

    else: # bytes_keeper is empty
        print("bytes_keeper is empty, cannot reconstruct video.")
    # --- End of video reconstruction attempt ---

    # assemble the video with the buffers
    import cv2
    import numpy as np

    # Sort frames by frame number
    frames = []
    for frame_num in sorted(value_keeper.keys()):
        buffer_y, buffer_cb, buffer_cr = value_keeper[frame_num]
        buffer_y_np = np.array(buffer_y, dtype=np.uint8)
        buffer_cb_np = np.array(buffer_cb, dtype=np.uint8)
        buffer_cr_np = np.array(buffer_cr, dtype=np.uint8)

        # Resize chroma planes to match luma plane
        buffer_cb_np = cv2.resize(buffer_cb_np, (buffer_y_np.shape[1], buffer_y_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        buffer_cr_np = cv2.resize(buffer_cr_np, (buffer_y_np.shape[1], buffer_y_np.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Merge Y, Cr, Cb for OpenCV (Y, Cr, Cb order)
        ycbcr_image = cv2.merge((buffer_y_np, buffer_cr_np, buffer_cb_np))
        bgr_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)
        frames.append(bgr_image)

    if frames:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
        for img in frames:
            out.write(img)
        out.release()
        print(f"Video saved as output.avi")
        
        
                
    # # assemble a frame and display it for the first frame
    # if len(value_keeper) > 0:
    #     frame = list(value_keeper.keys())[0]
        
    #     buffer_y, buffer_cb, buffer_cr = value_keeper[frame]
        
    #     # Assuming the buffers are numpy arrays or similar
    #     # You can use OpenCV or any other library to display the image
    #     import cv2
    #     import numpy as np
    #     from src.pyencoder.utils.video_reader import VideoReader
    #     # Convert Y, Cb, Cr buffers to numpy arrays
    #     buffer_y_np = np.array(buffer_y, dtype=np.uint8)
    #     buffer_cb_np = np.array(buffer_cb, dtype=np.uint8)
    #     buffer_cr_np = np.array(buffer_cr, dtype=np.uint8)

    #     # OpenCV uses Y, Cr, Cb order
    #     buffer_cb_np = cv2.resize(buffer_cb_np, (buffer_y_np.shape[1], buffer_y_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    #     buffer_cr_np = cv2.resize(buffer_cr_np, (buffer_y_np.shape[1], buffer_y_np.shape[0]), interpolation=cv2.INTER_LINEAR)
    #     ycbcr_image = cv2.merge((buffer_y_np, buffer_cb_np, buffer_cr_np))

    #     bgr_image = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2BGR)
    #     cv2.imwrite(f"frame_{frame}_{i}.png", bgr_image)
        
        
        