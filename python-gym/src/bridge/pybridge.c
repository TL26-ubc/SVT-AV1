#define PY_SSIZE_T_CLEAN
#include "pybridge.h"
#include "cb_registration.h"
#include "py_trampoline.h"

int (*get_deltaq_offset_cb)(unsigned sb_index, unsigned sb_org_x, unsigned sb_org_y, uint8_t sb_qindex,
                            uint16_t sb_final_blk_cnt, int32_t mi_row_start, int32_t mi_row_end, int32_t mi_col_start,
                            int32_t mi_col_end, int32_t tg_horz_boundary, int32_t tile_row, int32_t tile_col,
                            int32_t tile_rs_index, int32_t picture_number, uint8_t *buffer_y, uint8_t *buffer_cb,
                            uint8_t *buffer_cr, uint16_t sb_width, uint16_t sb_height, uint8_t encoder_bit_depth,
                            int32_t qindex, double beta, int32_t type, void *user);

void (*frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                          uint32_t origin_x, uint32_t origin_y, uint32_t stride_y, uint32_t stride_cb,
                          uint32_t stride_cr, uint32_t width, uint32_t height, void *user) = NULL;

int get_deltaq_offset_trampoline(
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
    int32_t picture_number,
    uint8_t *buffer_y,
    uint8_t *buffer_cb,
    uint8_t *buffer_cr,
    uint16_t sb_width,
    uint16_t sb_height,
    uint8_t encoder_bit_depth,
    int32_t qindex,
    double beta,
    int32_t type,
    void *user
) {
    int deltaq = 0;
    Callback *cb = &g_callbacks[CB_GET_DELTAQ_OFFSET];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
            cb->cb_fmt,  // Format string
            &deltaq,
            sb_index,           // I: unsigned
            sb_org_x,           // I: unsigned
            sb_org_y,           // I: unsigned
            sb_qindex,          // B: unsigned
            sb_final_blk_cnt,   // H: unsigned
            mi_row_start,       // i: int32_t
            mi_row_end,         // i: int32_t
            mi_col_start,       // i: int32_t
            mi_col_end,         // i: int32_t
            tg_horz_boundary,   // i: int32_t
            tile_row,           // i: int32_t
            tile_col,           // i: int32_t
            tile_rs_index,      // i: int32_t
            picture_number,     // i: int32_t
            buffer_y, sb_width, sb_height,       // M: uint8_t*, int, int
            buffer_cb, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            buffer_cr, sb_width / 2, sb_height / 2, // M: uint8_t*, int, int
            sb_width,           // I: uint16_t
            sb_height,          // I: uint16_t
            encoder_bit_depth,  // I: uint8_t
            qindex,             // i: int32_t
            beta,               // d: double
            type == 1           // b: int (converted to bool)
        );
    }
    return deltaq;
}

static PyObject *create_buffer_list(uint8_t *buffer, int width, int height, int stride, int origin_x, int origin_y) {
    if (!buffer || width <= 0 || height <= 0 || stride <= 0)
        return NULL;

    PyObject *list = PyList_New(height);
    if (!list)
        return NULL;

    for (int i = 0; i < height; ++i) {
        PyObject *row = PyList_New(width);
        if (!row) {
            Py_DECREF(list);
            return NULL;
        }
        int buf_row_start = (origin_y + i) * stride + origin_x;
        for (int j = 0; j < width; ++j) {
            PyObject *value = PyLong_FromLong((long)buffer[buf_row_start + j]);
            if (!value) {
                Py_DECREF(row);
                Py_DECREF(list);
                return NULL;
            }
            PyList_SetItem(row, j, value);
        }
        PyList_SetItem(list, i, row);
    }
    return list;
}

static void frame_feedback_trampoline(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr,
                                      uint32_t picture_number, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                                      uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height,
                                      void *user) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    // make python 1D arrays from the buffers
    PyObject *list_y  = create_buffer_list(buffer_y, width, height, stride_y, origin_x, origin_y);
    PyObject *list_cb = create_buffer_list(buffer_cb, width / 2, height / 2, stride_cb, origin_x / 2, origin_y / 2);
    PyObject *list_cr = create_buffer_list(buffer_cr, width / 2, height / 2, stride_cr, origin_x / 2, origin_y / 2);
    if (!list_y || !list_cb || !list_cr) {
        PyErr_SetString(PyExc_ValueError, "Failed to create Python lists from buffers");
        goto cleanup;
    }

    if (py_frame_feedback_obj && PyCallable_Check(py_frame_feedback_obj)) {
        PyObject *args = Py_BuildValue("(iiiOOO)",
                                       picture_number,
                                       width, // I: uint32_t
                                       height, // I: uint32_t
                                       list_y, // O: PyObject*
                                       list_cb, // O: PyObject*
                                       list_cr // O: PyObject*);
        );

        if (args) {
            PyObject *ret = PyObject_CallObject(py_frame_feedback_obj, args);
            Py_XDECREF(ret);
            Py_DECREF(args);
        }

        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    }

cleanup:
    Py_XDECREF(list_y);
    Py_XDECREF(list_cb);
    Py_XDECREF(list_cr);
    PyGILState_Release(gstate);
    // Release the GIL
}

void pybridge_set_callbacks(PyObject *get_deltaq_offset, PyObject *frame_feedback) {
    // Set QP offset callback
    Py_XINCREF(get_deltaq_offset);
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = get_deltaq_offset;
    get_deltaq_offset_cb     = get_deltaq_offset ? get_deltaq_offset_trampoline : NULL;

    // Set frame feedback callback
    Py_XINCREF(frame_feedback);
    Py_XDECREF(py_frame_feedback_obj);
    py_frame_feedback_obj = frame_feedback;
    frame_feedback_cb     = frame_feedback ? frame_feedback_trampoline : NULL;
}

void pybridge_clear(void) {
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = NULL;
    get_deltaq_offset_cb     = NULL;

    Py_XDECREF(py_frame_feedback_obj);
    py_frame_feedback_obj = NULL;
    frame_feedback_cb     = NULL;
}