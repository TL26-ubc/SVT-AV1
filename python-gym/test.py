import argparse
import os
from typing import List
import pandas as pd # Import pandas
import json # For potentially serializing complex data

import pyencoder

def get_deltaq_offset(
    sb_info_list: List[dict],
    sb_total_count: int,
    picture_number: int,
    frame_type: int
) -> List[int]:
    return [0]*(sb_total_count-1)

    print(f"Frame {picture_number}: width={width}, height={height}, bytes_used={bytes_used}")
    # update rl model
    
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
    
    global frame_counter
    if picture_number not in frame_counter:
        frame_counter[picture_number] = 0
    frame_counter[picture_number] += 1

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

    # print out all frame number counts if more than 1
    print("\nFrame Feedback Counts:")
    for frame_num, count in sorted(frame_counter.items()):
        if count > 1:
            print(f"Frame {frame_num}: {count} feedback calls")
            
    assert len(frame_counter) == 300
    # make sure the keys are 0 to 299
    assert all(k in frame_counter for k in range(300)), "Frame numbers should be 0 to 299"
    
    # make sure the keys are unique
    assert len(frame_counter) == len(set(frame_counter.keys())), "Frame numbers should be unique"
            
    print("check passed")
    # render the video from bytes_keeper
    # # --- Attempt to render video from collected bitstreams ---
    # if bytes_keeper:
    #     print("\nAttempting to reconstruct video from collected bitstreams...")

    #     # --- Retrieve video dimensions for IVF header ---
    #     video_width = 0
    #     video_height = 0
    #     # Try to get dimensions from excel_data_rows, which should be populated by frame_feedback
    #     if excel_data_rows:
    #         for row_data in sorted(excel_data_rows, key=lambda x: x.get("Frame Number", float('inf'))):
    #             if "Width (FrameFb)" in row_data and "Height (FrameFb)" in row_data:
    #                 video_width = row_data["Width (FrameFb)"]
    #                 video_height = row_data["Height (FrameFb)"]
    #                 if video_width > 0 and video_height > 0: # Ensure valid dimensions
    #                     break
        
    #     if video_width == 0 or video_height == 0:
    #         # Fallback: Try to infer from value_keeper if YUV data is there and consistent
    #         # This is less ideal as frame_feedback is the direct source for encoded dimensions
    #         print("Warning: Could not get dimensions from frame_feedback via excel_data_rows. Trying to infer from YUV buffers.")
    #         if value_keeper:
    #             first_frame_num = sorted(value_keeper.keys())[0]
    #             buffer_y, _, _ = value_keeper[first_frame_num]
    #             if buffer_y and isinstance(buffer_y, list) and len(buffer_y) > 0 and isinstance(buffer_y[0], list):
    #                 video_height = len(buffer_y)
    #                 video_width = len(buffer_y[0])
    #                 print(f"Inferred dimensions from YUV buffer: {video_width}x{video_height}")

    #     if video_width == 0 or video_height == 0:
    #         print("Error: Video dimensions (width/height) could not be determined. Cannot create IVF file.")
    #         # Set a flag or handle this error to skip ffmpeg processing if dimensions are missing
    #         can_reconstruct_video = False
    #     else:
    #         can_reconstruct_video = True
    #     # --- End of dimension retrieval ---

    #     if can_reconstruct_video:
    #         ivf_file = "reconstructed_av1_stream.ivf" # Changed extension
    #         output_video_file_mkv = "reconstructed_video_from_callbacks.mkv"
    #         output_video_file_mp4 = "reconstructed_video_from_callbacks.mp4"

    #         try:
    #             with open(ivf_file, "wb") as f:
    #                 # Write IVF File Header (32 bytes)
    #                 f.write(b'DKIF')  # Signature
    #                 f.write((0).to_bytes(2, 'little'))  # Version
    #                 f.write((32).to_bytes(2, 'little'))  # Header length
    #                 f.write(b'AV01')  # Codec FourCC
    #                 f.write(video_width.to_bytes(2, 'little'))
    #                 f.write(video_height.to_bytes(2, 'little'))
                    
    #                 # Frame rate and time scale (e.g., 30 fps)
    #                 # Using a common timebase like 1/1000 for PTS if actual rate is unknown
    #                 # Or, assume a fixed rate like 30fps. Let's use 30/1 for simplicity.
    #                 frame_rate_num = 30 
    #                 frame_rate_den = 1
    #                 f.write(frame_rate_num.to_bytes(4, 'little')) 
    #                 f.write(frame_rate_den.to_bytes(4, 'little')) 
                    
    #                 # Placeholder for number of frames, will be updated later
    #                 num_frames_ivf_pos = f.tell()
    #                 f.write((0).to_bytes(4, 'little'))  # Number of frames (placeholder)
    #                 f.write((0).to_bytes(4, 'little'))  # Unused

    #                 print("IVF Frame Data Summary (first 5 valid frames or fewer):")
    #                 pts_counter = 0 
    #                 frames_written_to_ivf = 0
    #                 MIN_VALID_FRAME_SIZE = 150 # Stricter minimum valid frame size

    #                 all_frames_from_keeper_modified = []
    #                 skipped_due_to_size_count = 0
    #                 skipped_frame_8 = False

    #                 for frame_num in sorted(bytes_keeper.keys()):
    #                     bitstream_data = bytes_keeper[frame_num]
    #                     original_len = len(bitstream_data)

    #                     if frame_num == 8:
    #                         skipped_frame_8 = True
    #                         if original_len > 2:
    #                             bitstream_data = bitstream_data[:-2]
    #                             print(f"Using trimmed Frame 8: Original size {original_len}, Trimmed size {len(bitstream_data)}")
    #                         else:
    #                             print(f"Frame 8 is too short ({original_len} bytes) to trim, using as is (will likely be skipped by size filter).")
    #                         # Update length for size check after potential trimming
    #                         current_len = len(bitstream_data) 
    #                     else:
    #                         current_len = original_len

    #                     if current_len >= MIN_VALID_FRAME_SIZE:
    #                         all_frames_from_keeper_modified.append((frame_num, bitstream_data))
    #                     else:
    #                         if frame_num != 8 : # Avoid double-counting if Frame 8 was already too small before trimming
    #                             print(f"Skipping Frame {frame_num} (size {current_len}B) - less than {MIN_VALID_FRAME_SIZE}B.")
    #                             skipped_due_to_size_count +=1
    #                         elif frame_num == 8 and original_len < MIN_VALID_FRAME_SIZE and current_len < MIN_VALID_FRAME_SIZE:
    #                             # If original Frame 8 was already too small and remained so after trimming (or wasn't trimmed)
    #                             print(f"Skipping trimmed Frame 8 (size {current_len}B) - less than {MIN_VALID_FRAME_SIZE}B.")
    #                             skipped_due_to_size_count +=1

    #                 if skipped_frame_8 and not any(f[0]==8 for f in all_frames_from_keeper_modified) and len(bytes_keeper.get(8, b'')) >= MIN_VALID_FRAME_SIZE:
    #                      # This case means Frame 8 was initially large enough, got trimmed, and the trimmed version was also large enough but somehow wasn't added.
    #                      # This should ideally not happen with current logic but is a safeguard print.
    #                      # Or, it means Frame 8 was large, trimmed, and then the trimmed version became too small.
    #                      if len(bytes_keeper.get(8,b'')[:-2]) < MIN_VALID_FRAME_SIZE and len(bytes_keeper.get(8, b'')) >= MIN_VALID_FRAME_SIZE:
    #                          print(f"Note: Trimmed Frame 8 was skipped as its new size {len(bytes_keeper.get(8,b'')[:-2])}B was less than {MIN_VALID_FRAME_SIZE}B.")
    #                          skipped_due_to_size_count +=1 # Count it if it was skipped due to trimming making it too small

    #                 print(f"Total frames skipped due to size: {skipped_due_to_size_count}")
    #                 frames_to_write = all_frames_from_keeper_modified
                                        
    #                 if not frames_to_write or frames_to_write[0][0] != 0:
    #                     print("Error: Frame 0 is missing, not selected due to size, or not first. Cannot proceed.")
    #                     # Check if Frame 0 specifically was skipped due to size
    #                     if 0 not in [f[0] for f in frames_to_write] and 0 in bytes_keeper and len(bytes_keeper[0]) < MIN_VALID_FRAME_SIZE:
    #                         print(f"Reason: Frame 0 (size {len(bytes_keeper[0])}B) was smaller than MIN_VALID_FRAME_SIZE ({MIN_VALID_FRAME_SIZE}B).")
    #                     can_reconstruct_video = False
    #                 else:
    #                     print(f"Preparing to write {len(frames_to_write)} frames (Frame 8 trimmed, min size {MIN_VALID_FRAME_SIZE}B) to IVF. Frames up to: {frames_to_write[-1][0]}")
    #                     pts_counter = 0
    #                     frames_written_to_ivf = 0

    #                 for frame_num, bitstream_data in frames_to_write:
    #                     frame_size = len(bitstream_data)
    #                     print(f"  Writing Frame {frame_num} (PTS {pts_counter}): {frame_size} bytes")
                        
    #                     f.write(frame_size.to_bytes(4, 'little'))
    #                     f.write(pts_counter.to_bytes(8, 'little')) 
                        
    #                     f.write(bitstream_data)
    #                     pts_counter += 1 # pts_counter will be 0 for the first frame
    #                     frames_written_to_ivf += 1
                    
    #                 # Go back and write the actual number of frames written (0 or 1)
    #                 f.seek(num_frames_ivf_pos)
    #                 f.write(frames_written_to_ivf.to_bytes(4, 'little'))
    #                 f.flush() # Ensure all data is written before getting size or closing
                
    #             print(f"AV1 stream packaged into IVF file: {ivf_file} ({frames_written_to_ivf} frames written, {os.path.getsize(ivf_file)} bytes)")

    #             # --- Add diagnostic print for the first valid frame's bitstream data ---
    #             if frames_to_write: # Check if there was any valid frame to write
    #                 first_valid_frame_num, first_valid_bitstream = frames_to_write[0]
    #                 print(f"First valid frame ({first_valid_frame_num}) bitstream (first 16 bytes hex):")
    #                 print(' '.join(f'{b:02x}' for b in first_valid_bitstream[:16]))
    #             # --- End of diagnostic print ---

    #             import subprocess
    #             # Attempt 1: Try to copy the AV1 stream directly from IVF into an MKV container
    #             ffmpeg_cmd_mkv = [
    #                 "ffmpeg",
    #                 "-y",
    #                 "-i", ivf_file, # Input is now the IVF file
    #                 "-c", "copy",      
    #                 output_video_file_mkv
    #             ]
    #             current_output_video_file = output_video_file_mkv

    #             print(f"Running ffmpeg command (Attempt 1 - copy from IVF to MKV): {' '.join(ffmpeg_cmd_mkv)}")
    #             process_mkv = subprocess.run(ffmpeg_cmd_mkv, capture_output=True, text=True)
                
    #             if process_mkv.returncode == 0:
    #                 print(f"Video successfully reconstructed (copied from IVF) and saved as {current_output_video_file}")
    #             else:
    #                 print(f"Attempt 1 (copy from IVF to MKV) failed (return code: {process_mkv.returncode}). FFmpeg stderr:\n{process_mkv.stderr}")
    #                 print("Attempting to re-encode to MP4 from IVF...")
                    
    #                 # Attempt 2: Re-encode to MP4 from IVF
    #                 ffmpeg_cmd_mp4 = [
    #                     "ffmpeg",
    #                     "-y",
    #                     "-i", ivf_file, # Input is now the IVF file
    #                     "-c:v", "libx264", 
    #                     output_video_file_mp4 
    #                 ]
    #                 current_output_video_file = output_video_file_mp4 
    #                 print(f"Running ffmpeg command (Attempt 2 - re-encode from IVF to MP4 defaults): {' '.join(ffmpeg_cmd_mp4)}")
    #                 process_mp4 = subprocess.run(ffmpeg_cmd_mp4, capture_output=True, text=True)

    #                 if process_mp4.returncode == 0:
    #                     print(f"Video successfully reconstructed (re-encoded from IVF) and saved as {current_output_video_file}")
    #                 else:
    #                     print(f"Attempt 2 (re-encode from IVF to MP4) also failed (return code: {process_mp4.returncode}):")
    #                     print("FFmpeg stdout:")
    #                     print(process_mp4.stdout)
    #                     print("FFmpeg stderr:")
    #                     print(process_mp4.stderr)
    #                     print(f"Please check if '{ivf_file}' is a valid IVF file and the AV1 data within is correct.")

    #         except FileNotFoundError:
    #              print(f"Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH.")
    #         except Exception as e:
    #             print(f"An error occurred during IVF creation or video reconstruction: {e}")
    #     elif not bytes_keeper: # This case was already there, kept for completeness
    #          print("No bitstream data was collected in bytes_keeper to reconstruct video.")
    #     # else: # This implies can_reconstruct_video is False due to missing dimensions
    #          # print("Video reconstruction skipped due to missing dimensions.") # Message already printed above

    # else: # bytes_keeper is empty
    #     print("bytes_keeper is empty, cannot reconstruct video.")
    # # --- End of video reconstruction attempt ---
