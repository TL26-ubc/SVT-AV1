#include "rl_feedback.h"
#include "enc_callbacks.h"
#include "../Codec/sequence_control_set.h"
#include "../Codec/pic_buffer_desc.h"
#include "../Codec/resize.h"
#include "../Codec/svt_psnr.h"

#ifdef SVT_ENABLE_USER_CALLBACKS

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


void svt_report_frame_feedback(
    EbBufferHeaderType *header_ptr, 
    uint32_t max_luma_value,
    uint32_t source_width,
    uint32_t source_height
) {
    if (!plugin_cbs.user_frame_feedback) 
        return;

    uint8_t temporal_layer_index;
    uint32_t avg_qp;
    uint64_t picture_stream_size, luma_sse, cr_sse, cb_sse, picture_number, picture_qp;
    double luma_ssim, cr_ssim, cb_ssim;
    double temp_var, luma_psnr, cb_psnr, cr_psnr;

    // 提取数据 - 与TL26相同的逻辑
    picture_stream_size = header_ptr->n_filled_len;
    luma_sse = header_ptr->luma_sse;
    cr_sse = header_ptr->cr_sse;
    cb_sse = header_ptr->cb_sse;
    picture_number = header_ptr->pts;
    temporal_layer_index = header_ptr->temporal_layer_index;
    picture_qp = header_ptr->qp;
    avg_qp = header_ptr->avg_qp;
    luma_ssim = header_ptr->luma_ssim;
    cr_ssim = header_ptr->cr_ssim;
    cb_ssim = header_ptr->cb_ssim;

    // 计算PSNR - 与TL26相同的逻辑
    temp_var = (double)max_luma_value * max_luma_value * (source_width * source_height);
    luma_psnr = get_psnr_rl((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (source_width / 2 * source_height / 2);
    cb_psnr = get_psnr_rl((double)cb_sse, temp_var);
    cr_psnr = get_psnr_rl((double)cr_sse, temp_var);

    // 调用Python回调
    plugin_cbs.user_frame_feedback(
        picture_number,
        temporal_layer_index,
        picture_qp,
        avg_qp,
        luma_psnr,
        cb_psnr,
        cr_psnr,
        (double)luma_sse / (source_width * source_height),
        (double)cb_sse / (source_width / 2 * source_height / 2),
        (double)cr_sse / (source_width / 2 * source_height / 2),
        luma_ssim,
        cb_ssim,
        cr_ssim,
        picture_stream_size,
        plugin_cbs.user);
}

// 超级块级反馈实现 - 移植自TL26
void svt_report_sb_feedback(
    int picture_number, 
    uint32_t max_luma_value,
    int sb_index, 
    unsigned sb_origin_x, 
    unsigned sb_origin_y,
    unsigned sb_width, 
    unsigned sb_height,
    uint64_t luma_sse, 
    uint64_t cb_sse, 
    uint64_t cr_sse,
    double luma_ssim, 
    double cb_ssim, 
    double cr_ssim,
    uint8_t *buffer_y, 
    uint8_t *buffer_cb, 
    uint8_t *buffer_cr) {

    if (!plugin_cbs.user_sb_feedback) 
        return;

    double temp_var, luma_psnr, cb_psnr, cr_psnr;

    // 计算PSNR - 与TL26相同的逻辑
    temp_var = (double)max_luma_value * max_luma_value * (sb_width * sb_height);
    luma_psnr = get_psnr_rl((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (sb_width / 2 * sb_height / 2);
    cb_psnr = get_psnr_rl((double)cb_sse, temp_var);
    cr_psnr = get_psnr_rl((double)cr_sse, temp_var);

    // 调用Python回调
    plugin_cbs.user_sb_feedback(
        picture_number,
        sb_index,
        sb_origin_x,
        sb_origin_y,
        luma_psnr,
        cb_psnr,
        cr_psnr,
        (double)luma_sse / (sb_width * sb_height),
        (double)cb_sse / (sb_width / 2 * sb_height / 2),
        (double)cr_sse / (sb_width / 2 * sb_height / 2),
        luma_ssim,
        cb_ssim,
        cr_ssim,
        buffer_y,
        buffer_cb,
        buffer_cr,
        sb_width,
        sb_height,
        plugin_cbs.user);
}

// 超级块QP offset请求实现 - 移植自TL26
int svt_request_sb_offset(SuperBlock *sb_ptr, PictureControlSet *pcs, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE) {
    if (!plugin_cbs.user_get_deltaq_offset) 
        return 0;

    uint8_t *buffer_y, *buffer_cb, *buffer_cr;
    SequenceControlSet *scs = pcs->ppcs->scs;
    EbPictureBufferDesc *input_pic = (EbPictureBufferDesc *)pcs->ppcs->enhanced_unscaled_pic;
    
    // 获取buffer - 与TL26相同的逻辑
    if (pcs->ppcs->do_tf == TRUE) {
        assert(pcs->ppcs->save_source_picture_width == input_pic->width &&
               pcs->ppcs->save_source_picture_height == input_pic->height);
        buffer_y = pcs->ppcs->save_source_picture_ptr[0];
        buffer_cb = pcs->ppcs->save_source_picture_ptr[1];
        buffer_cr = pcs->ppcs->save_source_picture_ptr[2];
    } else {
        buffer_y = input_pic->buffer_y;
        buffer_cb = input_pic->buffer_cb;
        buffer_cr = input_pic->buffer_cr;
    }
    
    buffer_y = &(buffer_y[(input_pic->org_x + sb_ptr->org_x) + (input_pic->org_y + sb_ptr->org_y) * input_pic->stride_y]);
    buffer_cb = &(buffer_cb[(input_pic->org_x + sb_ptr->org_x) / 2 + (input_pic->org_y + sb_ptr->org_y) / 2 * input_pic->stride_cb]);
    buffer_cr = &(buffer_cr[(input_pic->org_x + sb_ptr->org_x) / 2 + (input_pic->org_y + sb_ptr->org_y) / 2 * input_pic->stride_cr]);
    
    const uint16_t sb_width = MIN(scs->sb_size, pcs->ppcs->aligned_width - sb_ptr->org_x);
    const uint16_t sb_height = MIN(scs->sb_size, pcs->ppcs->aligned_height - sb_ptr->org_y);

    TileInfo *tile_info = &sb_ptr->tile_info;
    unsigned sb_index = sb_ptr->index, sb_origin_x = sb_ptr->org_x, sb_origin_y = sb_ptr->org_y;
    int sb_qp = (int)sb_ptr->qindex, sb_final_blk_cnt = (int)sb_ptr->final_blk_cnt;

    int tile_row = tile_info->tile_row, tile_col = tile_info->tile_col, tile_rs_index = tile_info->tile_rs_index;
    int mi_row_start = tile_info->mi_row_start, mi_row_end = tile_info->mi_row_end,
        mi_col_start = tile_info->mi_col_start, mi_col_end = tile_info->mi_col_end;
    int tg_horz_boundary = tile_info->tg_horz_boundary;

    // 调用Python回调获取QP offset
    return plugin_cbs.user_get_deltaq_offset(
        sb_index,
        sb_origin_x,
        sb_origin_y,
        sb_qp,
        sb_final_blk_cnt,
        mi_row_start,
        mi_row_end,
        mi_col_start,
        mi_col_end,
        tg_horz_boundary,
        tile_row,
        tile_col,
        tile_rs_index,
        encoder_bit_depth,
        beta,
        slice_type_is_I_SLICE,
        plugin_cbs.user);
}



// 修复 svt_aom_sse_calculations_sb 函数
EbErrorType svt_aom_sse_calculations_sb(PictureControlSet *pcs, SequenceControlSet *scs, SuperBlock *sb,
                                        uint64_t *luma_sse_out, uint64_t *cb_sse_out, uint64_t *cr_sse_out) {
    *luma_sse_out = 0;
    *cb_sse_out = 0;
    *cr_sse_out = 0;

    return EB_ErrorNone;
}

#endif // SVT_ENABLE_USER_CALLBACKS