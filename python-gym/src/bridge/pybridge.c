#define PY_SSIZE_T_CLEAN
#include "pybridge.h"
#include "cb_registration.h"
#include "py_trampoline.h"

/**
 * Create a Python list from a buffer.
 * The list will be a 2D list where each inner list represents a row of the buffer.
 */
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

void (*recv_picture_feedback)(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number) = NULL;

int (*get_deltaq_offset_cb)(SuperBlockInfo *sb_info_array, int *offset_array, uint32_t sb_count,
                            int32_t picture_number, int32_t frame_type, void *user);

void (*recv_frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                               uint32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                               uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height,
                               void *user) = NULL;

int get_deltaq_offset_trampoline(SuperBlockInfo *sb_info_array, int *offset_array, uint32_t sb_count,
                                int32_t picture_number, int32_t frame_type, void *user) {
    int deltaq = 0;
    PyObject *py_sb_list = NULL;
    PyObject *py_offset_list = NULL;
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Convert offset_array to a Python list of integers
    py_offset_list = PyList_New(sb_count);
    if (!py_offset_list) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list for offsets");
        goto cleanup;
    }
    for (uint32_t i = 0; i < sb_count; ++i) {
        PyObject *item = PyLong_FromLong(offset_array[i]);
        if (!item) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create Python integer for offset");
            goto cleanup;
        }
        PyList_SetItem(py_offset_list, i, item); // Steals reference
    }

    // Convert sb_info_array to a Python list of dictionaries
    py_sb_list = PyList_New(sb_count);
    if (!py_sb_list) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list for SuperBlockInfo");
        goto cleanup;
    }
    for (uint32_t i = 0; i < sb_count; ++i) {
        SuperBlockInfo *current_sb_info = &sb_info_array[i];
        PyObject *sb_dict = PyDict_New();
        if (!sb_dict) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create Python dictionary for SuperBlockInfo");
            goto cleanup;
        }

        PyObject *val_sb_org_x = PyLong_FromUnsignedLong(current_sb_info->sb_org_x);
        PyObject *val_sb_org_y = PyLong_FromUnsignedLong(current_sb_info->sb_org_y);
        PyObject *val_sb_width = PyLong_FromUnsignedLong(current_sb_info->sb_width);
        PyObject *val_sb_height = PyLong_FromUnsignedLong(current_sb_info->sb_height);
        PyObject *val_sb_qindex = PyLong_FromUnsignedLong(current_sb_info->sb_qindex);
        PyObject *val_beta = PyFloat_FromDouble(current_sb_info->beta);

        if (!val_sb_org_x || !val_sb_org_y || !val_sb_width || !val_sb_height || !val_sb_qindex || !val_beta) {
            Py_XDECREF(val_sb_org_x); Py_XDECREF(val_sb_org_y); Py_XDECREF(val_sb_width);
            Py_XDECREF(val_sb_height); Py_XDECREF(val_sb_qindex); Py_XDECREF(val_beta);
            Py_DECREF(sb_dict);
            PyErr_SetString(PyExc_MemoryError, "Failed to create Python objects for SuperBlockInfo fields");
            goto cleanup;
        }

        int set_item_failed = 0;
        if (PyDict_SetItemString(sb_dict, "sb_org_x", val_sb_org_x) < 0) set_item_failed = 1;
        if (PyDict_SetItemString(sb_dict, "sb_org_y", val_sb_org_y) < 0) set_item_failed = 1;
        if (PyDict_SetItemString(sb_dict, "sb_width", val_sb_width) < 0) set_item_failed = 1;
        if (PyDict_SetItemString(sb_dict, "sb_height", val_sb_height) < 0) set_item_failed = 1;
        if (PyDict_SetItemString(sb_dict, "sb_qindex", val_sb_qindex) < 0) set_item_failed = 1;
        if (PyDict_SetItemString(sb_dict, "beta", val_beta) < 0) set_item_failed = 1;

        Py_DECREF(val_sb_org_x); Py_DECREF(val_sb_org_y); Py_DECREF(val_sb_width);
        Py_DECREF(val_sb_height); Py_DECREF(val_sb_qindex); Py_DECREF(val_beta);

        if (set_item_failed) {
            Py_DECREF(sb_dict);
            PyErr_SetString(PyExc_RuntimeError, "Failed to set SuperBlockInfo fields in dict");
            goto cleanup;
        }
        PyList_SetItem(py_sb_list, i, sb_dict); // Steals reference to sb_dict
    }

    Callback *cb = &g_callbacks[CB_GET_DELTAQ_OFFSET];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
                      cb->cb_fmt, // Format string will be (OOuii)i from cb_registration.c
                      &deltaq,
                      py_sb_list,     // O: PyObject* for SuperBlockInfo list
                      py_offset_list, // O: PyObject* for offset list
                      sb_count,       // u: uint32_t
                      picture_number, // i: int32_t
                      frame_type      // i: int32_t
        );
    }

cleanup:
    Py_XDECREF(py_sb_list);
    Py_XDECREF(py_offset_list);
    PyGILState_Release(gstate);
    return deltaq;
}

void recv_frame_feedback_trampoline(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                                    u_int32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
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

    Callback *cb = &g_callbacks[CB_RECV_FRAME_FEEDBACK];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
                      cb->cb_fmt, // Format string
                      NULL,
                      picture_number, // I: uint32_t
                      bytes_used, // I: uint32_t
                      width, // I: uint32_t
                      height, // I: uint32_t
                      list_y, // O: PyObject*
                      list_cb, // O: PyObject*
                      list_cr // O: PyObject*
        );
    }
    
cleanup:
    Py_XDECREF(list_y);
    Py_XDECREF(list_cb);
    Py_XDECREF(list_cr);
    PyGILState_Release(gstate);
    // Release the GIL
}

void recv_picture_feedback_trampoline(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    // create a python bytes object from the bitstream
    PyObject *bitStreamObj = PyBytes_FromStringAndSize((const char *)bitStream, bitstream_size);
    if (!bitStreamObj) {
        PyErr_SetString(PyExc_ValueError, "Failed to create Python bytes object from bitstream");
        PyGILState_Release(gstate);
        return;
    }
    Callback *cb = &g_callbacks[CB_RECV_PICTURE_FEEDBACK];
    // Call the user callback with the bitstream data
    if (!cb->py_callable) {
        Py_DECREF(bitStreamObj);
        PyGILState_Release(gstate);
        return;
    }

    if (cb->py_callable) {
        py_trampoline(cb->py_callable,
                      cb->cb_fmt, // Format string
                      NULL,
                      bitStreamObj, // O: PyObject*
                      picture_number // I: uint32_t
        );
    }

    PyGILState_Release(gstate);
}

