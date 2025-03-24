#include "tl26_flags.h"
#include "tl26_request.h"
#include "tl26_python_thread.h"

int request_sb_offset(SuperBlock *sb_ptr, int picture_number, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE) {
#ifdef TL26_RL
    TileInfo *tile_info = &sb_ptr->tile_info;
    unsigned  sb_index = sb_ptr->index, sb_origin_x = sb_ptr->org_x, sb_origin_y = sb_ptr->org_y;
    int       sb_qp = (int)sb_ptr->qindex, sb_final_blk_cnt = (int)sb_ptr->final_blk_cnt;

    int tile_row = tile_info->tile_row, tile_col = tile_info->tile_col, tile_rs_index = tile_info->tile_rs_index;
    int mi_row_start = tile_info->mi_row_start, mi_row_end = tile_info->mi_row_end,
        mi_col_start = tile_info->mi_col_start, mi_col_end = tile_info->mi_col_end;
    int tg_horz_boundary = tile_info->tg_horz_boundary;

    int type = slice_type_is_I_SLICE ? 1 : 0;

    return submit_sb_offset_request(
        sb_index,
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
        picture_number,
        encoder_bit_depth,
        qindex,
        beta,
        type);
#else
    return -1;
#endif
}