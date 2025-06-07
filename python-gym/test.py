import argparse
import json  # For potentially serializing complex data
import os

import cv2
import numpy as np
import pandas as pd  # Import pandas
import pyencoder

global frame_counter, value_keeper, bytes_keeper, excel_data_rows  # Add excel_data_rows
frame_counter = {}
value_keeper = {}
bytes_keeper = {}
excel_data_rows = []  # Initialize a list to store data for Excel


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
    frame_data = _get_frame_data_dict(picture_number)
    frame_data["Frame Type (DeltaQ)"] = "INTRA" if frame_type == 1 else "INTER"  # type: ignore

    # Store a summary of sb_info_list. For example, count or first SB's info.
    # Avoid storing the whole list if it's very large.
    if sb_info_list:
        frame_data["Superblock Info Summary (DeltaQ)"] = f"{len(sb_info_list)} SBs received. First SB qindex: {sb_info_list[0].get('sb_qindex', 'N/A')}, beta: {sb_info_list[0].get('beta', 'N/A'):.2f}"  # type: ignore
    else:
        frame_data["Superblock Info Summary (DeltaQ)"] = "No SB info received"  # type: ignore

    print(
        f"Python: get_deltaq_offset called for Frame {picture_number}, Type: {frame_type}, Total SBs: {sb_total_count}"
    )
    if len(sb_info_list) != sb_total_count:
        raise RuntimeError("Python Error: List lengths do not match sb_total_count!")

    is_intra_frame = frame_type == 1  # Assuming 1 means I_SLICE

    calculated_offsets_summary = []
    offset_list_to_fill = [0] * sb_total_count

    for i in range(sb_total_count):
        current_sb_info = sb_info_list[i]

        # Extract available info from the current_sb_info dictionary
        sb_org_x = current_sb_info.get("sb_org_x", 0)
        sb_org_y = current_sb_info.get("sb_org_y", 0)
        sb_width = current_sb_info.get("sb_width", 0)
        sb_height = current_sb_info.get("sb_height", 0)
        sb_qindex = current_sb_info.get("sb_qindex", 50)  # SB's current QP
        print(current_sb_info)
        beta = current_sb_info.get("beta", 0.5)

        # --- Print received SuperBlockInfo for verification ---
        # if i == 0: # Print only for the first SB to reduce log spam
        # print(f"  SB Index {i}:")
        # print(f"    Raw SB Info: {current_sb_info}")
        # print(f"    Parsed - Position: ({sb_org_x},{sb_org_y}), Size: {sb_width}x{sb_height}")
        # print(f"    Parsed - SB_QP: {sb_qindex}, Beta: {beta:.4f}")
        # The old version had frame-level qindex and encoder_bit_depth. These are not
        # directly passed in the new (OOuii)i signature via sb_info_list.
        # If needed, the C bridge or SuperBlockInfo struct would need to be extended.
        # print(f"    Frame_QIndex (MISSING): unknown, Encoder Bit Depth (MISSING): unknown")

        # --- Example QP offset calculation logic (adapted) ---
        qp_offset = 0

        if beta > 0.8:  # High confidence from TPL, likely complex
            qp_offset = -2
        elif beta < 0.2:  # Low confidence, possibly smooth
            qp_offset = 2
        else:
            # Frame-level qindex is not available directly.
            # Using sb_qindex for this part of the logic.
            if sb_qindex > 45:  # If SB's current QP is already high
                qp_offset = -1
            elif sb_qindex < 25:  # If SB's current QP is low
                qp_offset = 1

        if is_intra_frame:
            qp_offset -= 1  # Often, intra blocks benefit from slightly lower QP

        # Example region-based adjustment (ensure you know video dimensions if using absolute coordinates)
        # This part would require video_width/height which are not passed.
        # video_width = 176 # Example for qCIF
        # video_height = 144 # Example for qCIF
        # center_x, center_y = sb_org_x + sb_width // 2, sb_org_y + sb_height // 2
        # if center_x < video_width / 3 and center_y < video_height / 3: # Top-left region
        #     qp_offset -= 1

        qp_offset = max(-5, min(5, qp_offset))  # Clamp offset to a reasonable range

        # print(f"    Decision: QP offset = {qp_offset}")

        offset_list_to_fill[i] = qp_offset
        if i < 5:  # Store first 5 calculated offsets as an example
            calculated_offsets_summary.append(qp_offset)

    frame_data["Calculated QP Offsets (First 5)"] = str(calculated_offsets_summary)  # type: ignore
    # print(f"Python: get_deltaq_offset completed for Frame {picture_number}.\n")
    return offset_list_to_fill  # Indicate success


def picture_feedback(bitstream: bytes, size: int, picture_number: int):
    frame_data = _get_frame_data_dict(picture_number)
    # Store cumulative size if called multiple times
    global bytes_keeper
    if picture_number in bytes_keeper:
        frame_data["Bitstream Size (PicFb)"] = frame_data.get(
            "Bitstream Size (PicFb)", 0
        ) + len(bitstream)
    else:
        frame_data["Bitstream Size (PicFb)"] = len(bitstream)

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

    # Clear data from previous runs if any
    excel_data_rows.clear()
    frame_counter.clear()
    value_keeper.clear()
    bytes_keeper.clear()  # Ensure bytes_keeper is cleared as a dictionary

    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset, picture_feedback=picture_feedback
    )
    pyencoder.run(
        input=args.file, pred_struct=1, rc=2, tbr=1000, enable_stat_report=True, 
        b="Output/output.ivf",  # Example output file for bitstream
    )
