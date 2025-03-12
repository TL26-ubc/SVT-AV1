#include "tl26_request.h"
#include "../../App/tl26_flags.h"

int request_sb_offset(SuperBlock *sb_ptr, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE) {
    if (PyCallable_Check(f_sb_send_offset_request)) {
        TileInfo *tile_info = &sb_ptr->tile_info;
        PyObject *args      = Py_BuildValue("iiiiiiiiiiiiiiidb",
                                       sb_ptr->index,
                                       sb_ptr->org_x,
                                       sb_ptr->org_y,
                                       sb_ptr->qindex,
                                       sb_ptr->final_blk_cnt,

                                       tile_info->mi_row_start,
                                       tile_info->mi_row_end,
                                       tile_info->mi_col_start,
                                       tile_info->mi_col_end,
                                       tile_info->tg_horz_boundary,
                                       tile_info->tile_row,
                                       tile_info->tile_col,
                                       tile_info->tile_rs_index,

                                       encoder_bit_depth,
                                       qindex,
                                       beta,
                                       slice_type_is_I_SLICE);
        PyObject *pValue    = PyObject_CallObject(f_sb_send_offset_request, args);
        if (pValue != NULL) {
            int offset = PyLong_AsLong(pValue);
            Py_DECREF(pValue);
            return offset;
        } else {
            PyErr_Print();
        }
        Py_DECREF(args);
    } else {
        PyErr_Print();
    }
    return -1;
}