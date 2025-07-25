/*
 * Copyright (c) 2024, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <arm_neon.h>
#include "definitions.h"
#include "cabac_context_model.h"
#include "common_utils.h"
#include "full_loop.h"
#include "mem_neon.h"
#include "motion_estimation.h" //svt_aom_downsample_2d_c()

void svt_av1_txb_init_levels_neon(const TranLow *const coeff, const int32_t width, const int32_t height,
                                  uint8_t *const levels) {
    const int32_t   stride = width + TX_PAD_HOR;
    const int32x4_t zeros  = vdupq_n_s32(0);
    int32_t         i      = 0;
    int32_t         j      = 0;
    uint8_t        *ls     = levels;
    const TranLow  *cf     = coeff;

    memset(levels - TX_PAD_TOP * stride, 0, sizeof(*levels) * TX_PAD_TOP * stride);
    memset(levels + stride * height, 0, sizeof(*levels) * (TX_PAD_BOTTOM * stride + TX_PAD_END));

    if (width == 4) {
        do {
            const int32x4_t  coeffA  = vld1q_s32(cf);
            const int32x4_t  coeffB  = vld1q_s32(cf + width);
            const int16x8_t  coeffAB = vcombine_s16(vqmovn_s32(coeffA), vqmovn_s32(coeffB));
            const int16x8_t  absAB   = vqabsq_s16(coeffAB);
            const int8x8_t   absABs  = vqmovn_s16(absAB);
            const int8x16_t  absAB8  = vcombine_s8(absABs, vreinterpret_s8_s32(vget_low_s32(zeros)));
            const uint8x16_t lsAB    = vreinterpretq_u8_s32(vzip1q_s32(vreinterpretq_s32_s8(absAB8), zeros));
            vst1q_u8(ls, lsAB);
            ls += (stride << 1);
            cf += (width << 1);
            i += 2;
        } while (i < height);
    } else if (width == 8) {
        do {
            const int32x4_t  coeffA  = vld1q_s32(cf);
            const int32x4_t  coeffB  = vld1q_s32(cf + 4);
            const int16x8_t  coeffAB = vcombine_s16(vqmovn_s32(coeffA), vqmovn_s32(coeffB));
            const int16x8_t  absAB   = vqabsq_s16(coeffAB);
            const uint8x16_t absAB8  = vreinterpretq_u8_s8(
                vcombine_s8(vqmovn_s16(absAB), vreinterpret_s8_s32(vget_low_s32(zeros))));
            vst1q_u8(ls, absAB8);
            ls += stride;
            cf += width;
            i += 1;
        } while (i < height);
    } else {
        do {
            j = 0;
            do {
                const int32x4_t  coeffA  = vld1q_s32(cf);
                const int32x4_t  coeffB  = vld1q_s32(cf + 4);
                const int32x4_t  coeffC  = vld1q_s32(cf + 8);
                const int32x4_t  coeffD  = vld1q_s32(cf + 12);
                const int16x8_t  coeffAB = vcombine_s16(vqmovn_s32(coeffA), vqmovn_s32(coeffB));
                const int16x8_t  coeffCD = vcombine_s16(vqmovn_s32(coeffC), vqmovn_s32(coeffD));
                const int16x8_t  absAB   = vqabsq_s16(coeffAB);
                const int16x8_t  absCD   = vqabsq_s16(coeffCD);
                const uint8x16_t absABCD = vreinterpretq_u8_s8(vcombine_s8(vqmovn_s16(absAB), vqmovn_s16(absCD)));
                vst1q_u8((ls + j), absABCD);
                j += 16;
                cf += 16;
            } while (j < width);
            *(int32_t *)(ls + width) = 0;
            ls += stride;
            i += 1;
        } while (i < height);
    }
}

/* get_4_nz_map_contexts_2d coefficients: */
static const DECLARE_ALIGNED(16, uint8_t, c_4_po_2d[2][16]) = {
    {0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21},
    {0, 11, 11, 11, 11, 11, 11, 11, 6, 6, 21, 21, 6, 21, 21, 21}};

/* get_4_nz_map_contexts_ver coefficients: */
static const DECLARE_ALIGNED(16, uint8_t, c_4_po_ver[16]) = {SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10};

/* get_8_coeff_contexts_2d coefficients:
if (height == 8) */
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_2d_8[2][16]) = {
    {0, 1, 6, 6, 21, 21, 21, 21, 1, 6, 6, 21, 21, 21, 21, 21},
    {6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21}};

/* if (height < 8) */
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_2d_l[2][16]) = {
    {0, 16, 6, 6, 21, 21, 21, 21, 16, 16, 6, 21, 21, 21, 21, 21},
    {16, 16, 21, 21, 21, 21, 21, 21, 16, 16, 21, 21, 21, 21, 21, 21}};

/* if (height > 8) */
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_2d_g[2][16]) = {
    {0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11},
    {6, 6, 21, 21, 21, 21, 21, 21, 6, 21, 21, 21, 21, 21, 21, 21}};

/* get_4_nz_map_contexts_ver coefficients: */
static const DECLARE_ALIGNED(16, uint8_t, c_8_po_ver[16]) = {SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 0,
                                                             SIG_COEF_CONTEXTS_2D + 5,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10,
                                                             SIG_COEF_CONTEXTS_2D + 10};

/* get_16n_coeff_contexts_2d coefficients:
real_width == real_height */
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_2d_e[4][16]) = {
    {0, 1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {1, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21}};

/* real_width > real_height */
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_2d_g[3][16]) = {
    {0, 16, 6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {16, 16, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {16, 16, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21}};

/* real_width < real_height */
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_2d_l[2][16]) = {
    {6, 6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21},
    {6, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21}};

/* get_16n_coeff_contexts_hor coefficients: */
static const DECLARE_ALIGNED(16, uint8_t, c_16_po_ver[16]) = {SIG_COEF_CONTEXTS_2D + 0,
                                                              SIG_COEF_CONTEXTS_2D + 5,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10,
                                                              SIG_COEF_CONTEXTS_2D + 10};

/* end of coefficients declaration area */

static inline uint8x16_t load_8bit_4x4_to_1_reg(const uint8_t *const src, const int byte_stride) {
    uint32x4_t v_data = vld1q_u32((uint32_t *)src);
    v_data            = vld1q_lane_u32((uint32_t *)(src + 1 * byte_stride), v_data, 1);
    v_data            = vld1q_lane_u32((uint32_t *)(src + 2 * byte_stride), v_data, 2);
    v_data            = vld1q_lane_u32((uint32_t *)(src + 3 * byte_stride), v_data, 3);

    return vreinterpretq_u8_u32(v_data);
}

static inline uint8x16_t load_8bit_8x2_to_1_reg(const uint8_t *const src, const int byte_stride) {
    uint64x2_t v_data = vld1q_u64((uint64_t *)src);
    v_data            = vld1q_lane_u64((uint64_t *)(src + 1 * byte_stride), v_data, 1);

    return vreinterpretq_u8_u64(v_data);
}

static inline uint8x16_t load_8bit_16x1_to_1_reg(const uint8_t *const src) { return vld1q_u8(src); }

static inline void load_levels_4x4x5(const uint8_t *const src, const int stride, const ptrdiff_t *const offsets,
                                     uint8x16_t *const level) {
    level[0] = load_8bit_4x4_to_1_reg(&src[1], stride);
    level[1] = load_8bit_4x4_to_1_reg(&src[stride], stride);
    level[2] = load_8bit_4x4_to_1_reg(&src[offsets[0]], stride);
    level[3] = load_8bit_4x4_to_1_reg(&src[offsets[1]], stride);
    level[4] = load_8bit_4x4_to_1_reg(&src[offsets[2]], stride);
}

static inline void load_levels_8x2x5(const uint8_t *const src, const int stride, const ptrdiff_t *const offsets,
                                     uint8x16_t *const level) {
    level[0] = load_8bit_8x2_to_1_reg(&src[1], stride);
    level[1] = load_8bit_8x2_to_1_reg(&src[stride], stride);
    level[2] = load_8bit_8x2_to_1_reg(&src[offsets[0]], stride);
    level[3] = load_8bit_8x2_to_1_reg(&src[offsets[1]], stride);
    level[4] = load_8bit_8x2_to_1_reg(&src[offsets[2]], stride);
}

static inline void load_levels_16x1x5(const uint8_t *const src, const int stride, const ptrdiff_t *const offsets,
                                      uint8x16_t *const level) {
    level[0] = load_8bit_16x1_to_1_reg(&src[1]);
    level[1] = load_8bit_16x1_to_1_reg(&src[stride]);
    level[2] = load_8bit_16x1_to_1_reg(&src[offsets[0]]);
    level[3] = load_8bit_16x1_to_1_reg(&src[offsets[1]]);
    level[4] = load_8bit_16x1_to_1_reg(&src[offsets[2]]);
}

static inline uint8x16_t get_coeff_contexts_kernel(uint8x16_t *const level) {
    const uint8x16_t const_3 = vdupq_n_u8(3);
    const uint8x16_t const_4 = vdupq_n_u8(4);
    uint8x16_t       count;

    count    = vminq_u8(level[0], const_3);
    level[1] = vminq_u8(level[1], const_3);
    level[2] = vminq_u8(level[2], const_3);
    level[3] = vminq_u8(level[3], const_3);
    level[4] = vminq_u8(level[4], const_3);
    count    = vaddq_u8(count, level[1]);
    count    = vaddq_u8(count, level[2]);
    count    = vaddq_u8(count, level[3]);
    count    = vaddq_u8(count, level[4]);

    count = vrshrq_n_u8(count, 1);
    count = vminq_u8(count, const_4);
    return count;
}

static inline void get_4_nz_map_contexts_2d(const uint8_t *levels, const int32_t height, const ptrdiff_t *const offsets,
                                            uint8_t *const coeff_contexts) {
    const int32_t    stride              = 4 + TX_PAD_HOR;
    const uint8x16_t pos_to_offset_large = vdupq_n_u8(21);

    uint8x16_t pos_to_offset = vld1q_u8((height == 4) ? c_4_po_2d[0] : c_4_po_2d[1]);

    uint8x16_t count;
    uint8x16_t level[5];
    uint8_t   *cc  = coeff_contexts;
    int        row = height;

    assert(!(height % 4));

    do {
        load_levels_4x4x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset);
        vst1q_u8(cc, count);
        pos_to_offset = pos_to_offset_large;
        levels += 4 * stride;
        cc += 16;
        row -= 4;
    } while (row);

    coeff_contexts[0] = 0;
}

static inline void get_8_coeff_contexts_2d(const uint8_t *levels, const int32_t height, const ptrdiff_t *const offsets,
                                           uint8_t *coeff_contexts) {
    const int32_t stride = 8 + TX_PAD_HOR;
    uint8_t      *cc     = coeff_contexts;
    uint8x16_t    count;
    uint8x16_t    level[5];
    uint8x16_t    pos_to_offset[3];
    int32_t       row = height;

    assert(!(height % 2));

    if (height == 8) {
        pos_to_offset[0] = vld1q_u8(c_8_po_2d_8[0]);
        pos_to_offset[1] = vld1q_u8(c_8_po_2d_8[1]);
    } else if (height < 8) {
        pos_to_offset[0] = vld1q_u8(c_8_po_2d_l[0]);
        pos_to_offset[1] = vld1q_u8(c_8_po_2d_l[1]);
    } else {
        pos_to_offset[0] = vld1q_u8(c_8_po_2d_g[0]);
        pos_to_offset[1] = vld1q_u8(c_8_po_2d_g[1]);
    }
    pos_to_offset[2] = vdupq_n_u8(21);

    do {
        load_levels_8x2x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset[0]);
        vst1q_u8(cc, count);
        pos_to_offset[0] = pos_to_offset[1];
        pos_to_offset[1] = pos_to_offset[2];
        levels += 2 * stride;
        cc += 16;
        row -= 2;
    } while (row);

    coeff_contexts[0] = 0;
}

static inline void get_16n_coeff_contexts_2d(const uint8_t *levels, const int32_t real_width, const int32_t real_height,
                                             const int32_t width, const int32_t height, const ptrdiff_t *const offsets,
                                             uint8_t *coeff_contexts) {
    const int32_t stride = width + TX_PAD_HOR;
    uint8_t      *cc     = coeff_contexts;
    int32_t       row    = height;
    uint8x16_t    pos_to_offset[5];
    uint8x16_t    pos_to_offset_large[3];
    uint8x16_t    count;
    uint8x16_t    level[5];
    int32_t       w;

    assert(!(width % 16));

    pos_to_offset_large[2] = vdupq_n_u8(21);
    if (real_width == real_height) {
        pos_to_offset[0] = vld1q_u8(c_16_po_2d_e[0]);
        pos_to_offset[1] = vld1q_u8(c_16_po_2d_e[1]);
        pos_to_offset[2] = vld1q_u8(c_16_po_2d_e[2]);
        pos_to_offset[3] = vld1q_u8(c_16_po_2d_e[3]);
        pos_to_offset[4] = pos_to_offset_large[0] = pos_to_offset_large[1] = pos_to_offset_large[2];
    } else if (real_width > real_height) {
        pos_to_offset[0] = vld1q_u8(c_16_po_2d_g[0]);
        pos_to_offset[1] = vld1q_u8(c_16_po_2d_g[1]);
        pos_to_offset[2] = pos_to_offset[3] = pos_to_offset[4] = vld1q_u8(c_16_po_2d_g[2]);
        pos_to_offset_large[0] = pos_to_offset_large[1] = pos_to_offset_large[2];
    } else { // real_width < real_height
        pos_to_offset[0] = pos_to_offset[1] = vdupq_n_u8(11);
        pos_to_offset[2]                    = vld1q_u8(c_16_po_2d_l[0]);
        pos_to_offset[3]                    = vld1q_u8(c_16_po_2d_l[1]);
        pos_to_offset[4]                    = pos_to_offset_large[2];
        pos_to_offset_large[0] = pos_to_offset_large[1] = vdupq_n_u8(11);
    }

    do {
        w = width;

        do {
            load_levels_16x1x5(levels, stride, offsets, level);
            count = get_coeff_contexts_kernel(level);
            count = vaddq_u8(count, pos_to_offset[0]);
            vst1q_u8(cc, count);
            levels += 16;
            cc += 16;
            w -= 16;
            pos_to_offset[0] = pos_to_offset_large[0];
        } while (w);

        pos_to_offset[0]       = pos_to_offset[1];
        pos_to_offset[1]       = pos_to_offset[2];
        pos_to_offset[2]       = pos_to_offset[3];
        pos_to_offset[3]       = pos_to_offset[4];
        pos_to_offset_large[0] = pos_to_offset_large[1];
        pos_to_offset_large[1] = pos_to_offset_large[2];
        levels += TX_PAD_HOR;
    } while (--row);

    coeff_contexts[0] = 0;
}

static inline void get_4_nz_map_contexts_hor(const uint8_t *levels, const int32_t height,
                                             const ptrdiff_t *const offsets, uint8_t *coeff_contexts) {
    const int32_t    stride                         = 4 + TX_PAD_HOR;
    const int32_t    sig_coef_contexts_2d_x4_051010 = (SIG_COEF_CONTEXTS_2D + ((SIG_COEF_CONTEXTS_2D + 5) << 8) +
                                                    ((SIG_COEF_CONTEXTS_2D + 10) << 16) +
                                                    ((SIG_COEF_CONTEXTS_2D + 10) << 24));
    const uint8x16_t pos_to_offset                  = vreinterpretq_u8_u32(vdupq_n_u32(sig_coef_contexts_2d_x4_051010));

    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    row = height;

    assert(!(height % 4));

    do {
        load_levels_4x4x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset);
        vst1q_u8(coeff_contexts, count);
        levels += 4 * stride;
        coeff_contexts += 16;
        row -= 4;
    } while (row);
}

static inline void get_4_nz_map_contexts_ver(const uint8_t *levels, const int32_t height,
                                             const ptrdiff_t *const offsets, uint8_t *coeff_contexts) {
    const int32_t    stride              = 4 + TX_PAD_HOR;
    const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);

    uint8x16_t pos_to_offset = vld1q_u8(c_4_po_ver);

    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    row = height;

    assert(!(height % 4));

    do {
        load_levels_4x4x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset);
        vst1q_u8(coeff_contexts, count);
        pos_to_offset = pos_to_offset_large;
        levels += 4 * stride;
        coeff_contexts += 16;
        row -= 4;
    } while (row);
}

static inline void get_8_coeff_contexts_hor(const uint8_t *levels, const int32_t height, const ptrdiff_t *const offsets,
                                            uint8_t *coeff_contexts) {
    const int32_t stride = 8 + TX_PAD_HOR;

    const uint8x16_t pos_to_offset = vld1q_u8(c_8_po_ver);

    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    row = height;

    assert(!(height % 2));

    do {
        load_levels_8x2x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset);
        vst1q_u8(coeff_contexts, count);
        levels += 2 * stride;
        coeff_contexts += 16;
        row -= 2;
    } while (row);
}

static inline void get_8_coeff_contexts_ver(const uint8_t *levels, const int32_t height, const ptrdiff_t *const offsets,
                                            uint8_t *coeff_contexts) {
    const int32_t    stride              = 8 + TX_PAD_HOR;
    const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);

    uint8x16_t pos_to_offset = vcombine_u8(vdup_n_u8(SIG_COEF_CONTEXTS_2D + 0), vdup_n_u8(SIG_COEF_CONTEXTS_2D + 5));

    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    row = height;

    assert(!(height % 2));

    do {
        load_levels_8x2x5(levels, stride, offsets, level);
        count = get_coeff_contexts_kernel(level);
        count = vaddq_u8(count, pos_to_offset);
        vst1q_u8(coeff_contexts, count);
        pos_to_offset = pos_to_offset_large;
        levels += 2 * stride;
        coeff_contexts += 16;
        row -= 2;
    } while (row);
}

static inline void get_16n_coeff_contexts_hor(const uint8_t *levels, const int32_t width, const int32_t height,
                                              const ptrdiff_t *const offsets, uint8_t *coeff_contexts) {
    const int32_t stride = width + TX_PAD_HOR;

    const uint8x16_t pos_to_offset_large = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);

    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    w, row = height;

    assert(!(width % 16));

    do {
        uint8x16_t pos_to_offset = vld1q_u8(c_16_po_ver);

        w = width;
        do {
            load_levels_16x1x5(levels, stride, offsets, level);
            count = get_coeff_contexts_kernel(level);
            count = vaddq_u8(count, pos_to_offset);
            vst1q_u8(coeff_contexts, count);
            pos_to_offset = pos_to_offset_large;
            levels += 16;
            coeff_contexts += 16;
            w -= 16;
        } while (w);

        levels += TX_PAD_HOR;
    } while (--row);
}

static inline void get_16n_coeff_contexts_ver(const uint8_t *levels, const int32_t width, const int32_t height,
                                              const ptrdiff_t *const offsets, uint8_t *coeff_contexts) {
    const int32_t stride = width + TX_PAD_HOR;

    uint8x16_t pos_to_offset[3];
    uint8x16_t count;
    uint8x16_t level[5];
    int32_t    w, row = height;

    assert(!(width % 16));

    pos_to_offset[0] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 0);
    pos_to_offset[1] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 5);
    pos_to_offset[2] = vdupq_n_u8(SIG_COEF_CONTEXTS_2D + 10);

    do {
        w = width;
        do {
            load_levels_16x1x5(levels, stride, offsets, level);
            count = get_coeff_contexts_kernel(level);
            count = vaddq_u8(count, pos_to_offset[0]);
            vst1q_u8(coeff_contexts, count);
            levels += 16;
            coeff_contexts += 16;
            w -= 16;
        } while (w);

        pos_to_offset[0] = pos_to_offset[1];
        pos_to_offset[1] = pos_to_offset[2];
        levels += TX_PAD_HOR;
    } while (--row);
}

void svt_av1_get_nz_map_contexts_neon(const uint8_t *const levels, const int16_t *const scan, const uint16_t eob,
                                      TxSize tx_size, const TxClass tx_class, int8_t *const coeff_contexts) {
    const int32_t last_idx = eob - 1;
    if (!last_idx) {
        coeff_contexts[0] = 0;
        return;
    }
    uint8_t *const coefficients = (uint8_t *const)coeff_contexts;

    const int32_t real_width  = tx_size_wide[tx_size];
    const int32_t real_height = tx_size_high[tx_size];
    const int32_t width       = get_txb_wide(tx_size);
    const int32_t height      = get_txb_high(tx_size);
    const int32_t stride      = width + TX_PAD_HOR;
    ptrdiff_t     offsets[3];

    /* coeff_contexts must be 16 byte aligned. */
    assert(!((intptr_t)coeff_contexts & 0xf));

    if (tx_class == TX_CLASS_2D) {
        offsets[0] = 0 * stride + 2;
        offsets[1] = 1 * stride + 1;
        offsets[2] = 2 * stride + 0;

        if (width == 4) {
            get_4_nz_map_contexts_2d(levels, height, offsets, coefficients);
        } else if (width == 8) {
            get_8_coeff_contexts_2d(levels, height, offsets, coefficients);
        } else {
            get_16n_coeff_contexts_2d(levels, real_width, real_height, width, height, offsets, coefficients);
        }
    } else if (tx_class == TX_CLASS_HORIZ) {
        offsets[0] = 2;
        offsets[1] = 3;
        offsets[2] = 4;
        if (width == 4) {
            get_4_nz_map_contexts_hor(levels, height, offsets, coefficients);
        } else if (width == 8) {
            get_8_coeff_contexts_hor(levels, height, offsets, coefficients);
        } else {
            get_16n_coeff_contexts_hor(levels, width, height, offsets, coefficients);
        }
    } else { // TX_CLASS_VERT
        offsets[0] = 2 * stride;
        offsets[1] = 3 * stride;
        offsets[2] = 4 * stride;
        if (width == 4) {
            get_4_nz_map_contexts_ver(levels, height, offsets, coefficients);
        } else if (width == 8) {
            get_8_coeff_contexts_ver(levels, height, offsets, coefficients);
        } else {
            get_16n_coeff_contexts_ver(levels, width, height, offsets, coefficients);
        }
    }

    const int32_t bwl = get_txb_bwl_tab[tx_size];
    const int32_t pos = scan[last_idx];
    if (last_idx <= (height << bwl) / 8) {
        coeff_contexts[pos] = 1;
    } else if (last_idx <= (height << bwl) / 4) {
        coeff_contexts[pos] = 2;
    } else {
        coeff_contexts[pos] = 3;
    }
}

static inline uint8x8_t compute_sum(uint8x16_t in, uint8x16_t prev_in) {
    uint16x8_t sum = vpaddlq_u8(in);
    sum            = vpadalq_u8(sum, prev_in);

    return vqrshrn_n_u16(sum, 2);
}

void svt_aom_downsample_2d_neon(uint8_t *input_samples, // input parameter, input samples Ptr
                                uint32_t input_stride, // input parameter, input stride
                                uint32_t input_area_width, // input parameter, input area width
                                uint32_t input_area_height, // input parameter, input area height
                                uint8_t *decim_samples, // output parameter, decimated samples Ptr
                                uint32_t decim_stride, // input parameter, output stride
                                uint32_t decim_step) // input parameter, decimation amount in pixels
{
    uint32_t input_stripe_stride = input_stride * decim_step;
    uint8_t *in_ptr              = input_samples;
    uint8_t *out_ptr             = decim_samples;
    uint32_t width_align16       = input_area_width - (input_area_width % 16);

    if (decim_step == 2) {
        in_ptr += input_stride;
        for (uint32_t vert_idx = 1; vert_idx < input_area_height; vert_idx += 2) {
            uint8_t *prev_in_line           = in_ptr - input_stride;
            uint32_t decim_horizontal_index = 0;

            for (uint32_t horiz_idx = 1; horiz_idx < width_align16; horiz_idx += 16) {
                uint8x16_t prev_in  = vld1q_u8(prev_in_line + horiz_idx - 1);
                uint8x16_t in       = vld1q_u8(in_ptr + horiz_idx - 1);
                uint8x8_t  sum_epu8 = compute_sum(in, prev_in);
                vst1_u8(out_ptr + decim_horizontal_index, sum_epu8);
                decim_horizontal_index += 8;
            }

            // complement when input_area_width is not multiple of 16
            if (width_align16 < input_area_width) {
                DECLARE_ALIGNED(16, uint8_t, tmp_buf[8]);
                uint8x16_t prev_in  = vld1q_u8(prev_in_line + width_align16);
                uint8x16_t in       = vld1q_u8(in_ptr + width_align16);
                uint8x8_t  sum_epu8 = compute_sum(in, prev_in);
                int        count    = (input_area_width - width_align16) >> 1;
                vst1_u8(tmp_buf, sum_epu8);
                memcpy(out_ptr + decim_horizontal_index, tmp_buf, count * sizeof(uint8_t));
            }

            in_ptr += input_stripe_stride;
            out_ptr += decim_stride;
        }
    } else if (decim_step == 4) {
        in_ptr += 2 * input_stride;
        for (uint32_t vertical_index = 2; vertical_index < input_area_height; vertical_index += 4) {
            uint8_t *prev_in_line           = in_ptr - input_stride;
            uint32_t decim_horizontal_index = 0;

            for (uint32_t horiz_idx = 2; horiz_idx < width_align16; horiz_idx += 16) {
                uint8x16_t prev_in  = vld1q_u8(prev_in_line + horiz_idx - 1);
                uint8x16_t in       = vld1q_u8(in_ptr + horiz_idx - 1);
                uint8x8_t  sum_epu8 = compute_sum(in, prev_in);
                sum_epu8            = vuzp1_u8(sum_epu8, sum_epu8);
                store_u8_4x1(out_ptr + decim_horizontal_index, sum_epu8);
                decim_horizontal_index += 4;
            }

            // complement when input_area_width is not multiple of 16
            if (width_align16 < input_area_width) {
                uint8x16_t prev_in  = vld1q_u8(prev_in_line + width_align16 + 1);
                uint8x16_t in       = vld1q_u8(in_ptr + width_align16 + 1);
                uint8x8_t  sum_epu8 = compute_sum(in, prev_in);
                sum_epu8            = vuzp1_u8(sum_epu8, sum_epu8);
                int      count      = (input_area_width - width_align16) >> 2;
                uint32_t tmp        = vget_lane_u32(vreinterpret_u32_u8(sum_epu8), 0);
                memcpy(out_ptr + decim_horizontal_index, &tmp, count * sizeof(uint8_t));
            }

            in_ptr += input_stripe_stride;
            out_ptr += decim_stride;
        }
    } else {
        // fallback for other decimation step values
        svt_aom_downsample_2d_c(
            input_samples, input_stride, input_area_width, input_area_height, decim_samples, decim_stride, decim_step);
    }
}

#if CLN_REMOVE_MODE_INFO
void svt_copy_mi_map_grid_neon(MbModeInfo **mi_grid_ptr, uint32_t mi_stride, uint8_t num_rows, uint8_t num_cols) {
    MbModeInfo *target = mi_grid_ptr[0];
#else
void svt_copy_mi_map_grid_neon(ModeInfo **mi_grid_ptr, uint32_t mi_stride, uint8_t num_rows, uint8_t num_cols) {
    ModeInfo *target = mi_grid_ptr[0];
#endif
    if (num_cols == 1) {
        for (uint8_t mi_y = 0; mi_y < num_rows; mi_y++) {
            const int32_t mi_idx = 0 + mi_y * mi_stride;
            // width is 1 block (corresponds to block width 4)
            mi_grid_ptr[mi_idx] = target;
        }
    } else if (num_cols == 2) {
        const uint64x2_t target_sse = vdupq_n_u64((uint64_t)target);
        for (uint8_t mi_y = 0; mi_y < num_rows; mi_y++) {
            const int32_t mi_idx = 0 + mi_y * mi_stride;
            // width is 2 blocks, so can copy 2 at once (corresponds to block width 8)
            vst1q_u64((uint64_t *)&mi_grid_ptr[mi_idx], target_sse);
        }
    } else {
        const uint64x2_t target_avx = vdupq_n_u64((uint64_t)target);
        for (uint8_t mi_y = 0; mi_y < num_rows; mi_y++) {
            for (uint8_t mi_x = 0; mi_x < num_cols; mi_x += 4) {
                const int32_t mi_idx = mi_x + mi_y * mi_stride;
                // width is >=4 blocks, so can copy 4 at once; (corresponds to block width >=16).
                // All blocks >= 16 have widths that are divisible by 16, so it is ok to copy 4 blocks at once
                vst1q_u64((uint64_t *)&mi_grid_ptr[mi_idx + 0], target_avx);
                vst1q_u64((uint64_t *)&mi_grid_ptr[mi_idx + 2], target_avx);
            }
        }
    }
}
