#include "app_bridge.h"
#include "rl_feedback.h"

void svt_report_frame_feedback_bridge(
    EbBufferHeaderType *header_ptr, 
    uint32_t max_luma_value,
    uint32_t source_width,
    uint32_t source_height
) {
    svt_report_frame_feedback(
        header_ptr, 
        max_luma_value,
        source_width,
        source_height
    );
    return;
}