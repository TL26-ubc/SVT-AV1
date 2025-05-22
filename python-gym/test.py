import pyencoder

def get_deltaq_offset(
    sb_index: int, sb_org_x: int, sb_org_y: int, sb_qindex: int, sb_final_blk_cnt: int,
    mi_row_start: int, mi_row_end: int, mi_col_start: int, mi_col_end: int,
    tg_horz_boundary: int, tile_row: int, tile_col: int, tile_rs_index: int,
    encoder_bit_depth: int, beta: float, is_intra: bool,
) -> int:
    """QP offset回调 - 与之前TL26接口保持一致"""
    print(f"QP offset request for SB {sb_index} at ({sb_org_x}, {sb_org_y})")
    # 这里可以调用你的RL模型
    return 0  # 返回QP offset

def frame_feedback(
    picture_number: int, temporal_layer_index: int, qp: int, avg_qp: int,
    luma_psnr: float, cb_psnr: float, cr_psnr: float,
    mse_y: float, mse_u: float, mse_v: float,
    luma_ssim: float, cb_ssim: float, cr_ssim: float,
    picture_stream_size: int
):
    """帧级反馈回调 - 与之前TL26接口保持一致"""
    print(f"Frame {picture_number}: PSNR={luma_psnr:.2f}, bits={picture_stream_size}")
    # 这里可以更新你的RL模型

def sb_feedback(
    picture_number: int, sb_index: int, sb_origin_x: int, sb_origin_y: int,
    luma_psnr: float, cb_psnr: float, cr_psnr: float,
    mse_y: float, mse_u: float, mse_v: float,
    luma_ssim: float, cb_ssim: float, cr_ssim: float,
    buffer_y: list, buffer_cb: list, buffer_cr: list
):
    """超级块级反馈回调 - 与之前TL26接口保持一致"""
    print(f"SB {sb_index} at ({sb_origin_x}, {sb_origin_y}): PSNR={luma_psnr:.2f}")
    # buffer_y, buffer_cb, buffer_cr 是与TL26相同格式的列表
    # 这里可以处理超级块级的反馈


pyencoder.register_callbacks(
    get_deltaq_offset=get_deltaq_offset,
    frame_feedback=frame_feedback,
    sb_feedback=sb_feedback
)

pyencoder.run(
    input="../../playground/akiyo_qcif.y4m",
    preset=8,
    crf=30,
    enable_stat_report=True
)


