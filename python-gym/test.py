import argparse
import os

import pyencoder


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
    print(len(buffer_y), "buffer_y")
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

    print(f"RL Model - Frame {picture_number}, SB {sb_index}:")
    print(f"  Position: ({sb_org_x},{sb_org_y}), Size: {sb_width}x{sb_height}")
    print(f"  QP: {sb_qindex}, QIndex: {qindex}, Beta: {beta:.4f}")
    print(f"  Tile: ({tile_row},{tile_col}), Type: {'INTRA' if is_intra else 'INTER'}")
    print(f"  Avg Luma: {avg_luma:.1f}, Texture: {texture_complexity:.1f}")

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

    print(f"  Decision: QP offset = {qp_offset}")
    print()

    return qp_offset


def frame_feedback(
    picture_number: int,
    temporal_layer_index: int,
    qp: int,
    avg_qp: int,
    luma_psnr: float,
    cb_psnr: float,
    cr_psnr: float,
    mse_y: float,
    mse_u: float,
    mse_v: float,
    luma_ssim: float,
    cb_ssim: float,
    cr_ssim: float,
    picture_stream_size: int,
):

    print(f"Frame {picture_number}: PSNR={luma_psnr:.2f}, bits={picture_stream_size}")
    # update rl model


def sb_feedback(
    picture_number: int,
    sb_index: int,
    sb_origin_x: int,
    sb_origin_y: int,
    luma_psnr: float,
    cb_psnr: float,
    cr_psnr: float,
    mse_y: float,
    mse_u: float,
    mse_v: float,
    luma_ssim: float,
    cb_ssim: float,
    cr_ssim: float,
    buffer_y: list,
    buffer_cb: list,
    buffer_cr: list,
):
    print(f"SB {sb_index} at ({sb_origin_x}, {sb_origin_y}): PSNR={luma_psnr:.2f}")
    # process sb feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)

    args = parser.parse_args()
    print(args)
    pyencoder.register_callbacks(
        get_deltaq_offset=get_deltaq_offset,
        frame_feedback=frame_feedback,
        sb_feedback=sb_feedback,
    )
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)
