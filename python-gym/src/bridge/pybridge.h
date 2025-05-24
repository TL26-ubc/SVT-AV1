#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>

extern int (*get_deltaq_offset_cb)(
    unsigned sb_index,
    unsigned sb_org_x,
    unsigned sb_org_y,
    uint8_t sb_qindex,
    uint16_t sb_final_blk_cnt,
    int32_t mi_row_start,
    int32_t mi_row_end,
    int32_t mi_col_start,
    int32_t mi_col_end,
    int32_t tg_horz_boundary,
    int32_t tile_row,
    int32_t tile_col,
    int32_t tile_rs_index,
    int32_t picture_number,      
    uint8_t *buffer_y,           
    uint8_t *buffer_cb,          
    uint8_t *buffer_cr,          
    uint16_t sb_width,           
    uint16_t sb_height,          
    uint8_t encoder_bit_depth,
    int32_t qindex,              
    double beta,
    int32_t type,                
    void* user);


extern void (*recv_frame_feedback_cb)(
    int picture_number,
    int temporal_layer_index,
    int qp,
    int avg_qp,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    int picture_stream_size,
    void *user);

extern void (*recv_sb_feedback_cb)(
    int picture_number,
    unsigned sb_index,
    unsigned sb_origin_x,
    unsigned sb_origin_y,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    uint8_t *buffer_y,
    uint8_t *buffer_cb,
    uint8_t *buffer_cr,
    uint16_t sb_width,
    uint16_t sb_height,
    void *user);

int get_deltaq_offset_trampoline(
    unsigned sb_index,
    unsigned sb_org_x,
    unsigned sb_org_y,
    uint8_t sb_qindex,
    uint16_t sb_final_blk_cnt,
    int32_t mi_row_start,
    int32_t mi_row_end,
    int32_t mi_col_start,
    int32_t mi_col_end,
    int32_t tg_horz_boundary,
    int32_t tile_row,
    int32_t tile_col,
    int32_t tile_rs_index,
    int32_t picture_number,
    uint8_t *buffer_y,
    uint8_t *buffer_cb,
    uint8_t *buffer_cr,
    uint16_t sb_width,
    uint16_t sb_height,
    uint8_t encoder_bit_depth,
    int32_t qindex,
    double beta,
    int32_t type,
    void *user
);

void recv_frame_feedback_trampoline(
    int picture_number,
    int temporal_layer_index,
    int qp,
    int avg_qp,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    int picture_stream_size,
    void *user
);

void recv_sb_feedback_trampoline(
    int picture_number,
    unsigned sb_index,
    unsigned sb_origin_x,
    unsigned sb_origin_y,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    uint8_t *buffer_y,
    uint8_t *buffer_cb,
    uint8_t *buffer_cr,
    uint16_t sb_width,
    uint16_t sb_height,
    void *user
);

#endif /* PYBRIDGE_H */
