#include <stdlib.h>
#include "../Lib/Codec/pcs.h"

void report_sb_feedback(int picture_number, uint32_t max_luma_value,
    int sb_index, unsigned sb_origin_x, unsigned sb_origin_y,
    unsigned sb_width, unsigned sb_height,
    uint64_t luma_sse, uint64_t cb_sse, uint64_t cr_sse,
    double luma_ssim, double cb_ssim, double cr_ssim,
    uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr);