#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#include <Python.h>
#include <stdint.h>
#include <stdbool.h>

extern int (*get_deltaq_offset_cb)(
    unsigned sb_index,
    unsigned sb_org_x,
    unsigned sb_org_y,
    uint8_t sb_qindex,
    uint16_t sb_final_blk_cnt,
    int32_t mi_row_start,
    int32_t mi_row_end,
    int32_t mi_col_start,
    int32_t mi_col_end,
    int32_t tg_horz_boundary,
    int32_t tile_row,
    int32_t tile_col,
    int32_t tile_rs_index,
    uint8_t encoder_bit_depth,
    double beta,
    bool is_intra, 
    void* user);

void pybridge_set_cb(PyObject *callable);
void pybridge_clear(void);

#endif /* PYBRIDGE_H */
