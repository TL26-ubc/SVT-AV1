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

void svt_report_encoded_frame(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                              u_int32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                              uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height) {
    if (!plugin_cbs.user_frame_feedback)
        return;

    // Call the user callback with the encoded frame data
    plugin_cbs.user_frame_feedback(buffer_y,
                                   buffer_cb,
                                   buffer_cr,
                                   picture_number,
                                   bytes_used,
                                   origin_x,
                                   origin_y,
                                   stride_y,
                                   stride_cb,
                                   stride_cr,
                                   width,
                                   height,
                                   plugin_cbs.user);
}

#endif // SVT_ENABLE_USER_CALLBACKS