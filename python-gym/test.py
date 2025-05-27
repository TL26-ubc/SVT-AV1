import argparse
import os

import pyencoder

global frame_counter, value_keeper, bytes_keeper
frame_counter = {}
value_keeper = {}
bytes_keeper = {}

def get_deltaq_offset(
    sb_index: int,
    sb_org_x: int,
    sb_org_y: int,
    sb_qindex: int,
    sb_final_blk_cnt: int,
    mi_row_start: int,
    mi_row_end: int,
    mi_col_start: int,
    mi_col_end: int,
    tg_horz_boundary: int,
    tile_row: int,
    tile_col: int,
    tile_rs_index: int,
    picture_number: int,
    buffer_y: list,
    buffer_cb: list,
    buffer_cr: list,
    sb_width: int,
    sb_height: int,
    encoder_bit_depth: int,
    qindex: int,
    beta: float,
    is_intra: bool,
) -> int:
    if buffer_y and len(buffer_y) > 0:
        total_pixels = sb_width * sb_height
        luma_sum = sum(sum(row) for row in buffer_y)
        avg_luma = luma_sum / total_pixels

        luma_variance = (
            sum(sum((pixel - avg_luma) ** 2 for pixel in row) for row in buffer_y)
            / total_pixels
        )
        texture_complexity = luma_variance**0.5
    else:
        avg_luma = 128
        texture_complexity = 0

    # print(f"RL Model - Frame {picture_number}, SB {sb_index}:")
    # print(f"  Position: ({sb_org_x},{sb_org_y}), Size: {sb_width}x{sb_height}")
    # print(f"  QP: {sb_qindex}, QIndex: {qindex}, Beta: {beta:.4f}")
    # print(f"  Tile: ({tile_row},{tile_col}), Type: {'INTRA' if is_intra else 'INTER'}")
    # print(f"  Avg Luma: {avg_luma:.1f}, Texture: {texture_complexity:.1f}")

    qp_offset = 0

    if texture_complexity < 10:
        if avg_luma > 200:
            qp_offset = 3
        elif avg_luma < 50:
            qp_offset = -1
    elif texture_complexity > 50:
        qp_offset = -2

    if is_intra:
        qp_offset -= 1

    center_x, center_y = sb_org_x + sb_width // 2, sb_org_y + sb_height // 2
    if center_x < 320 and center_y < 240:
        qp_offset -= 1

    qp_offset = max(-5, min(5, qp_offset))

    # print(f"  Decision: QP offset = {qp_offset}")
    # print()

    return qp_offset


def frame_feedback(
    picture_number: int,
    bytes_used: int,
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
    
    

def picture_feedback(
    bitstream: bytes,
    size: int,
    picture_number: int
):
    length = len(bitstream)
    global bytes_keeper
    bytes_keeper[picture_number] = bitstream

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)

    args = parser.parse_args()
    print(args)
    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset,
        frame_feedback=frame_feedback,
        picture_feedback=picture_feedback
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)

            
    # sum up the bytes used for all frames
    total_bytes = sum(len(bytes_keeper.values()))
    print(f"Total bytes used: {total_bytes}")
            
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
        
        
        