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

    // PSNR calculation 
    temp_var = (double)max_luma_value * max_luma_value * (source_width * source_height);
    luma_psnr = get_psnr_rl((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (source_width / 2 * source_height / 2);
    cb_psnr = get_psnr_rl((double)cb_sse, temp_var);
    cr_psnr = get_psnr_rl((double)cr_sse, temp_var);

    // callbacks
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

    temp_var = (double)max_luma_value * max_luma_value * (sb_width * sb_height);
    luma_psnr = get_psnr_rl((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (sb_width / 2 * sb_height / 2);
    cb_psnr = get_psnr_rl((double)cb_sse, temp_var);
    cr_psnr = get_psnr_rl((double)cr_sse, temp_var);

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


#endif // SVT_ENABLE_USER_CALLBACKS