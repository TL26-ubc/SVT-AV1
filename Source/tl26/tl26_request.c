#include "tl26_flags.h"
#include "tl26_request.h"
#include "tl26_python_thread.h"
#include "../Lib/Codec/sequence_control_set.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

int request_sb_offset(SuperBlock *sb_ptr, PictureControlSet *pcs, int encoder_bit_depth, int qindex, double beta,
                      bool slice_type_is_I_SLICE) {
#ifdef TL26_RL
    u_int8_t            *buffer_y, *buffer_cb, *buffer_cr;
    SequenceControlSet *scs = pcs->ppcs->scs;
    EbPictureBufferDesc *input_pic = (EbPictureBufferDesc *)pcs->ppcs->enhanced_unscaled_pic;
    if (pcs->ppcs->do_tf == TRUE) {
        assert(pcs->ppcs->save_source_picture_width == input_pic->width &&
               pcs->ppcs->save_source_picture_height == input_pic->height);
        buffer_y  = pcs->ppcs->save_source_picture_ptr[0];
        buffer_cb = pcs->ppcs->save_source_picture_ptr[1];
        buffer_cr = pcs->ppcs->save_source_picture_ptr[2];
    } else {
        buffer_y  = input_pic->buffer_y;
        buffer_cb = input_pic->buffer_cb;
        buffer_cr = input_pic->buffer_cr;
    }
    buffer_y = &(buffer_y[(input_pic->org_x + sb_ptr->org_x) + (input_pic->org_y + sb_ptr->org_y) * input_pic->stride_y]);
    buffer_cb = &(buffer_cb[(input_pic->org_x + sb_ptr->org_x) / 2 + (input_pic->org_y + sb_ptr->org_y) / 2 * input_pic->stride_cb]);
    buffer_cr = &(buffer_cr[(input_pic->org_x + sb_ptr->org_x) / 2 + (input_pic->org_y + sb_ptr->org_y) / 2 * input_pic->stride_cr]);
    const uint16_t sb_width  = MIN(scs->sb_size, pcs->ppcs->aligned_width  - sb_ptr->org_x);
    const uint16_t sb_height = MIN(scs->sb_size, pcs->ppcs->aligned_height - sb_ptr->org_y);

    TileInfo *tile_info = &sb_ptr->tile_info;
    unsigned  sb_index = sb_ptr->index, sb_origin_x = sb_ptr->org_x, sb_origin_y = sb_ptr->org_y;
    int       sb_qp = (int)sb_ptr->qindex, sb_final_blk_cnt = (int)sb_ptr->final_blk_cnt;

    int tile_row = tile_info->tile_row, tile_col = tile_info->tile_col, tile_rs_index = tile_info->tile_rs_index;
    int mi_row_start = tile_info->mi_row_start, mi_row_end = tile_info->mi_row_end,
        mi_col_start = tile_info->mi_col_start, mi_col_end = tile_info->mi_col_end;
    int tg_horz_boundary = tile_info->tg_horz_boundary;

    int type = slice_type_is_I_SLICE ? 1 : 0;

    return submit_sb_offset_request(sb_index,
                                    sb_origin_x,
                                    sb_origin_y,
                                    sb_qp,
                                    sb_final_blk_cnt,
                                    mi_row_start,
                                    mi_row_end,
                                    mi_col_start,
                                    mi_col_end,
                                    tg_horz_boundary,
                                    tile_row,
                                    tile_col,
                                    tile_rs_index,
                                    pcs->picture_number,
                                    buffer_y,
                                    buffer_cb,
                                    buffer_cr,
                                    sb_width,
                                    sb_height,
                                    encoder_bit_depth,
                                    qindex,
                                    beta,
                                    type);
#else
    return -1;
#endif
}