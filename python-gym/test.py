import argparse
import os

import pyencoder

global frame_counter, value_keeper
frame_counter = {}
value_keeper = {}

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
    print(f"Python: get_deltaq_offset called for Frame {picture_number}, Type: {frame_type}, Total SBs: {sb_total_count}")
    if len(sb_info_list) != sb_total_count or len(offset_list_to_fill) != sb_total_count:
        print("Python Error: List lengths do not match sb_total_count!")
        # Optionally raise an error or handle appropriately
        return -1 # Indicate an error

    is_intra_frame = (frame_type == 1) # Assuming 1 means I_SLICE

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

    print(f"Python: get_deltaq_offset completed for Frame {picture_number}.\n")
    return 0 # Indicate success


def frame_feedback(
    picture_number: int,
    width: int,
    height: int,
    buffer_y: list,
    buffer_cb: list,
    buffer_cr: list,
):
    print(f"Frame {picture_number}: width={width}, height={height}, ")
    # update rl model
    
    global frame_counter
    if picture_number not in frame_counter:
        frame_counter[picture_number] = 0
    frame_counter[picture_number] += 1
    
    global value_keeper
    value_keeper[picture_number] = (buffer_y, buffer_cb, buffer_cr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)

    args = parser.parse_args()
    print(args)
    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset,
        frame_feedback=frame_feedback,
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)
    
    for frame, count in sorted(frame_counter.items()):
        if count != 2:
            print(f"Frame {frame} - count: {count}")
            
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
        
        
        