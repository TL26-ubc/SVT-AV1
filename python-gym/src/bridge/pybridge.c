#define PY_SSIZE_T_CLEAN
#include "pybridge.h"
#include "cb_registration.h"
#include "py_trampoline.h"

int (*get_deltaq_offset_cb)(
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

void (*recv_frame_feedback_cb)(
    int,
    int,
    int,
    int,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    int,
    void*) = NULL;

void (*recv_sb_feedback_cb)(
    int,
    unsigned,
    unsigned,
    unsigned,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    uint8_t*,
    uint8_t*,
    uint8_t*,
    uint16_t,
    uint16_t,
    void*) = NULL;

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
) {
    int deltaq = 0;
    Callback *cb = &g_callbacks[CB_GET_DELTAQ_OFFSET];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
            cb->cb_fmt,  // Format string
            &deltaq,
            sb_index,           // I: unsigned
            sb_org_x,           // I: unsigned
            sb_org_y,           // I: unsigned
            sb_qindex,          // B: unsigned
            sb_final_blk_cnt,   // H: unsigned
            mi_row_start,       // i: int32_t
            mi_row_end,         // i: int32_t
            mi_col_start,       // i: int32_t
            mi_col_end,         // i: int32_t
            tg_horz_boundary,   // i: int32_t
            tile_row,           // i: int32_t
            tile_col,           // i: int32_t
            tile_rs_index,      // i: int32_t
            picture_number,     // i: int32_t
            buffer_y, sb_width, sb_height,       // M: uint8_t*, int, int
            buffer_cb, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            buffer_cr, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            sb_width,           // I: uint16_t
            sb_height,          // I: uint16_t
            encoder_bit_depth,  // I: uint8_t
            qindex,             // i: int32_t
            beta,               // d: double
            type == 1           // b: int (converted to bool)
        );
    }
    return deltaq;
}

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
) {
    Callback *cb = &g_callbacks[CB_RECV_FRAME_FEEDBACK];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
            cb->cb_fmt,
            NULL,
            picture_number,        // i: int
            temporal_layer_index,  // i: int
            qp,                    // i: int
            avg_qp,                // i: int
            luma_psnr,             // d: double
            cb_psnr,               // d: double
            cr_psnr,               // d: double
            mse_y,                 // d: double
            mse_u,                 // d: double
            mse_v,                 // d: double
            luma_ssim,             // d: double
            cb_ssim,               // d: double
            cr_ssim,               // d: double
            picture_stream_size    // i: int
        );
    }
}

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
) {
    Callback *cb = &g_callbacks[CB_RECV_SB_FEEDBACK];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
            cb->cb_fmt,
            NULL,
            picture_number,         // I: unsigned
            sb_index,               // i: int (promoted unsigned)
            sb_origin_x,            // i: int (promoted unsigned)
            sb_origin_y,            // i: int (promoted unsigned)
            luma_psnr,              // d: double
            cb_psnr,                // d: double
            cr_psnr,                // d: double
            mse_y,                  // d: double
            mse_u,                  // d: double
            mse_v,                  // d: double
            luma_ssim,              // d: double
            cb_ssim,                // d: double
            cr_ssim,                // d: double
            buffer_y, sb_width, sb_height,       // M: uint8_t*, int, int
            buffer_cb, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            buffer_cr, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            sb_width,              // H: uint16_t
            sb_height              // H: uint16_t
        );
    }
}
