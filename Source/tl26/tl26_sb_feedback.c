#include "tl26_flags.h"
#include "tl26_sb_feedback.h"
#include "tl26_python_thread.h"
#include "tl26_utils.h"

#ifdef TL26_RL
void report_sb_feedback(int picture_number, uint32_t max_luma_value,
    int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
    unsigned sb_width, unsigned sb_height,
    uint64_t luma_sse, uint64_t cb_sse, uint64_t cr_sse,
    double luma_ssim, double cb_ssim, double cr_ssim,
    uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr) {

    double   temp_var, luma_psnr, cb_psnr, cr_psnr;

    temp_var = (double)max_luma_value * max_luma_value * (sb_width * sb_height);
    luma_psnr = get_psnr_tl26((double)luma_sse, temp_var);

    temp_var = (double)max_luma_value * max_luma_value * (sb_width / 2 * sb_height / 2);
    cb_psnr = get_psnr_tl26((double)cb_sse, temp_var);
    cr_psnr = get_psnr_tl26((double)cr_sse, temp_var);

    submit_sb_feedback_request(picture_number,
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
        sb_height);
}
#else
void report_sb_feedback(int picture_number, uint32_t max_luma_value,
    int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
    unsigned sb_width, unsigned ,
    uint64_t luma_sse, uint64_t cb_sse, uint64_t cr_sse,
    double luma_ssim, double cb_ssim, double cr_ssim) {
    (void)picture_number;
    (void)max_luma_value;
    (void)sb_index;
    (void)sb_origin_x;
    (void)sb_origin_y;
    (void)sb_width;
    (void)sb_height;
    (void)luma_sse;
    (void)sb_height;
    (void)luma_sse;
    (void)cb_sse;
    (void)cr_sse;
    (void)luma_ssim;
    (void)cb_ssim;
    (void)cr_ssim;
}
#endif
