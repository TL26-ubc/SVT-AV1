#ifndef EB_RL_FEEDBACK_H
#define EB_RL_FEEDBACK_H

#include "EbSvtAv1Enc.h"
#include "EbSvtAv1ErrorCodes.h"
#include "../Codec/coding_unit.h"
#include "../Codec/block_structures.h"
#include "../Codec/pcs.h"
#include "../Lib/Codec/sequence_control_set.h"

#ifdef SVT_ENABLE_USER_CALLBACKS

static inline double get_psnr_rl(double sse, double max) {
    double psnr;
    if (sse == 0)
        psnr = 10 * log10(max / 0.1);
    else
        psnr = 10 * log10(max / sse);
    return psnr;
}

void svt_report_picture_feedback(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number);

#endif // SVT_ENABLE_USER_CALLBACKS

#endif // EB_RL_FEEDBACK_H