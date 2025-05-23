#ifndef EB_RL_FEEDBACK_H
#define EB_RL_FEEDBACK_H

#include "EbSvtAv1Enc.h"
#include "EbSvtAv1ErrorCodes.h"
#include "../Codec/coding_unit.h"
#include "../Codec/block_structures.h"
#include "../Codec/pcs.h"
// #include "../App/app_config.h"
#include "../Lib/Codec/sequence_control_set.h"


#ifdef SVT_ENABLE_USER_CALLBACKS

// 工具函数：计算PSNR
static inline double get_psnr_rl(double sse, double max) {
    double psnr;
    if (sse == 0)
        psnr = 10 * log10(max / 0.1);
    else
        psnr = 10 * log10(max / sse);
    return psnr;
}

// 帧级反馈
void svt_report_frame_feedback(
    EbBufferHeaderType *header_ptr, 
    uint32_t max_luma_value,
    uint32_t source_width,
    uint32_t source_height
);

// 超级块级反馈
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
    uint8_t *buffer_cr);

// 超级块QP offset请求
int svt_request_sb_offset(SuperBlock *sb_ptr, PictureControlSet *pcs, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE);


EbErrorType svt_aom_ssim_calculations_sb(PictureControlSet *pcs, SequenceControlSet *scs, SuperBlock *sb, Bool free_memory,
                                         double *luma_ssim_out, double *cb_ssim_out, double *cr_ssim_out);

// SSE计算函数 - 新增
EbErrorType svt_aom_sse_calculations_sb(PictureControlSet *pcs, SequenceControlSet *scs, SuperBlock *sb,
                                        uint64_t *luma_sse_out, uint64_t *cb_sse_out, uint64_t *cr_sse_out);

#endif // SVT_ENABLE_USER_CALLBACKS

#endif // EB_RL_FEEDBACK_H