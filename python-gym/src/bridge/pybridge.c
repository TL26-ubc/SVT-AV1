#define PY_SSIZE_T_CLEAN
#include "pybridge.h"

static PyObject *py_get_deltaq_offset_obj = NULL;
static PyObject *py_frame_feedback_obj    = NULL;

static PyObject *create_python_list_from_buffer(uint8_t *buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer or dimensions");
        return NULL;
    }

    PyObject *list = PyList_New(height);
    if (!list)
        return NULL;

    for (int i = 0; i < height; i++) {
        PyObject *row = PyList_New(width);
        if (!row) {
            Py_DECREF(list);
            return NULL;
        }

        for (int j = 0; j < width; j++) {
            int       index = i * width + j;
            PyObject *value = PyLong_FromLong((long)buffer[index]);
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

int (*get_deltaq_offset_cb)(unsigned sb_index, unsigned sb_org_x, unsigned sb_org_y, uint8_t sb_qindex,
                            uint16_t sb_final_blk_cnt, int32_t mi_row_start, int32_t mi_row_end, int32_t mi_col_start,
                            int32_t mi_col_end, int32_t tg_horz_boundary, int32_t tile_row, int32_t tile_col,
                            int32_t tile_rs_index, int32_t picture_number, uint8_t *buffer_y, uint8_t *buffer_cb,
                            uint8_t *buffer_cr, uint16_t sb_width, uint16_t sb_height, uint8_t encoder_bit_depth,
                            int32_t qindex, double beta, int32_t type, void *user);

void (*frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                          uint32_t origin_x, uint32_t origin_y, uint32_t stride_y, uint32_t stride_cb,
                          uint32_t stride_cr, uint32_t width, uint32_t height, void *user) = NULL;

// QP offset callback trampoline
static int get_deltaq_offset_trampoline(unsigned sb_index, unsigned sb_org_x, unsigned sb_org_y, uint8_t sb_qindex,
                                        uint16_t sb_final_blk_cnt, int32_t mi_row_start, int32_t mi_row_end,
                                        int32_t mi_col_start, int32_t mi_col_end, int32_t tg_horz_boundary,
                                        int32_t tile_row, int32_t tile_col, int32_t tile_rs_index,
                                        int32_t picture_number, uint8_t *buffer_y, uint8_t *buffer_cb,
                                        uint8_t *buffer_cr, uint16_t sb_width, uint16_t sb_height,
                                        uint8_t encoder_bit_depth, int32_t qindex, double beta, int32_t type,
                                        void *user) {
    int              deltaq = 0; // Default
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (py_get_deltaq_offset_obj && PyCallable_Check(py_get_deltaq_offset_obj)) {
        PyObject *py_buffer_y  = create_python_list_from_buffer(buffer_y, sb_width, sb_height);
        PyObject *py_buffer_cb = create_python_list_from_buffer(buffer_cb, sb_width / 2, sb_height / 2);
        PyObject *py_buffer_cr = create_python_list_from_buffer(buffer_cr, sb_width / 2, sb_height / 2);

        if (py_buffer_y && py_buffer_cb && py_buffer_cr) {
            PyObject *args = Py_BuildValue("(IIIBHiiiiiiiiiOOOHHBidO)",
                                           sb_index, // I: unsigned
                                           sb_org_x, // I: unsigned
                                           sb_org_y, // I: unsigned
                                           sb_qindex, // B: uint8_t
                                           sb_final_blk_cnt, // H: uint16_t
                                           mi_row_start, // i: int32_t
                                           mi_row_end, // i: int32_t
                                           mi_col_start, // i: int32_t
                                           mi_col_end, // i: int32_t
                                           tg_horz_boundary, // i: int32_t
                                           tile_row, // i: int32_t
                                           tile_col, // i: int32_t
                                           tile_rs_index, // i: int32_t
                                           picture_number, // i: int32_t
                                           py_buffer_y, // O: PyObject*
                                           py_buffer_cb, // O: PyObject*
                                           py_buffer_cr, // O: PyObject*
                                           sb_width, // H: uint16_t
                                           sb_height, // H: uint16_t
                                           encoder_bit_depth, // B: uint8_t
                                           qindex, // i: int32_t
                                           beta, // d: double
                                           PyBool_FromLong(type == 1) // O: bool (is_intra)
            );

            if (!args) {
                PyErr_Print();
                goto cleanup;
            }

            PyObject *ret = PyObject_CallObject(py_get_deltaq_offset_obj, args);
            Py_DECREF(args);

            if (!ret) {
                PyErr_Print();
                goto cleanup;
            }

            if (PyLong_Check(ret)) {
                deltaq = (int)PyLong_AsLong(ret);
            } else {
                PyErr_Format(PyExc_TypeError, "QP callback must return int, got %.200s", Py_TYPE(ret)->tp_name);
                PyErr_Print();
            }

            Py_XDECREF(ret);
        }

    cleanup:
        Py_XDECREF(py_buffer_y);
        Py_XDECREF(py_buffer_cb);
        Py_XDECREF(py_buffer_cr);
    }

    PyGILState_Release(gstate);
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