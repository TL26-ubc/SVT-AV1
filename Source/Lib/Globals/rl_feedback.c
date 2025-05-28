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

void svt_report_picture_feedback(uint8_t *bitstream, uint32_t bitstream_size, uint32_t picture_number) {
    if (!plugin_cbs.user_picture_feedback)
        return;

    // Call the user callback with the picture feedback data
    plugin_cbs.user_picture_feedback(bitstream, bitstream_size, picture_number, plugin_cbs.user);
}

#endif // SVT_ENABLE_USER_CALLBACKS