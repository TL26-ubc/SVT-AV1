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
    encoder_bit_depth: int,
    beta: float,
    is_intra: bool,
) -> int:
   
    print(f"QP offset request for SB {sb_index} at ({sb_org_x}, {sb_org_y})")
    #call rl model?
    return 0  # return QP offset

def frame_feedback(
    picture_number: int, temporal_layer_index: int, qp: int, avg_qp: int,
    luma_psnr: float, cb_psnr: float, cr_psnr: float,
    mse_y: float, mse_u: float, mse_v: float,
    luma_ssim: float, cb_ssim: float, cr_ssim: float,
    picture_stream_size: int
):
   
    print(f"Frame {picture_number}: PSNR={luma_psnr:.2f}, bits={picture_stream_size}")
    # update rl model

def sb_feedback(
    picture_number: int, sb_index: int, sb_origin_x: int, sb_origin_y: int,
    luma_psnr: float, cb_psnr: float, cr_psnr: float,
    mse_y: float, mse_u: float, mse_v: float,
    luma_ssim: float, cb_ssim: float, cr_ssim: float,
    buffer_y: list, buffer_cb: list, buffer_cr: list
):
    print(f"SB {sb_index} at ({sb_origin_x}, {sb_origin_y}): PSNR={luma_psnr:.2f}")
    # process sb feedback



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input video file", required=True)

    args = parser.parse_args()
    print(args)
    pyencoder.register_callbacks(get_deltaq_offset=get_deltaq_offset)
    pyencoder.run(input=args.file, rc=True, enable_stat_report=True)
