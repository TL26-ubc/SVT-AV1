#include "tl26_flags.h"
#include "tl26_request.h"

int request_sb_offset(SuperBlock *sb_ptr, int encoder_bit_depth, int qindex, double beta, bool slice_type_is_I_SLICE) {
    PyObject *f_sb_send_offset_request = get_f_sb_send_offset_request();
    if (PyCallable_Check(f_sb_send_offset_request)) {

        // PyGILState_STATE state = PyGILState_Ensure();
        TileInfo *tile_info = &sb_ptr->tile_info;
        unsigned  sb_index = sb_ptr->index, sb_origin_x = sb_ptr->org_x, sb_origin_y = sb_ptr->org_y;
        int       sb_qp = (int)sb_ptr->qindex, sb_final_blk_cnt = (int)sb_ptr->final_blk_cnt;

        int tile_row = tile_info->tile_row, tile_col = tile_info->tile_col, tile_rs_index = tile_info->tile_rs_index;
        int mi_row_start = tile_info->mi_row_start, mi_row_end = tile_info->mi_row_end,
            mi_col_start = tile_info->mi_col_start, mi_col_end = tile_info->mi_col_end;
        int tg_horz_boundary = tile_info->tg_horz_boundary;

        int type = slice_type_is_I_SLICE ? 1 : 0;

        PyObject *ptests = Py_BuildValue(
            "i", // this is telling the function to expect 14 arguments for int and double
            1);
        PyObject *argstest = Py_BuildValue(
            "ii", // this is telling the function to expect 14 arguments for int and double
            1,1);

        PyObject *args = Py_BuildValue("IIIiiiiiiiiiiiidi",
                                       sb_index,
                                       sb_origin_x,
                                       sb_origin_y,
                                       sb_qp,
                                       sb_final_blk_cnt, // sb
                                       mi_row_start,
                                       mi_row_end,
                                       mi_col_start,
                                       mi_col_end,
                                       tg_horz_boundary,
                                       tile_row,
                                       tile_col,
                                       tile_rs_index, // tile
                                       encoder_bit_depth,
                                       qindex,
                                       beta,
                                       type);


        // PyObject *args      = Py_BuildValue("IIIiiiiiiiiiiiidb",
        //                                sb_ptr->index, // unsigned
        //                                sb_ptr->org_x, // unsigned
        //                                sb_ptr->org_y, // unsigned
        //                                (int) sb_ptr->qindex, // uint8_t
        //                                (int) sb_ptr->final_blk_cnt, // uint16_t

        //                                tile_info->mi_row_start, // int32_t
        //                                tile_info->mi_row_end, // int32_t
        //                                tile_info->mi_col_start, // int32_t
        //                                tile_info->mi_col_end, // int32_t
        //                                tile_info->tg_horz_boundary, // int32_t
        //                                tile_info->tile_row, // int32_t
        //                                tile_info->tile_col, // int32_t
        //                                tile_info->tile_rs_index, // int32_t

        //                                encoder_bit_depth, // int
        //                                qindex, // int
        //                                beta, // double
        //                                slice_type_is_I_SLICE);
        PyObject *pValue = PyObject_CallObject(f_sb_send_offset_request, args);
        if (pValue != NULL) {
            int offset = PyLong_AsLong(pValue);
            Py_DECREF(pValue);
            // PyGILState_Release(state);
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