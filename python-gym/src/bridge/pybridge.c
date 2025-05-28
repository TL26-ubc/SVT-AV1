#define PY_SSIZE_T_CLEAN
#include "pybridge.h"
#include "cb_registration.h"
#include "py_trampoline.h"

void (*recv_picture_feedback)(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number,
                              void *user) = NULL; 

int *(*get_deltaq_offset_cb)(SuperBlockInfo *sb_info_array, uint32_t sb_count, int32_t picture_number, 
                            int32_t frame_type, void *user) = NULL;

void (*recv_frame_feedback_cb)(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                               uint32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                               uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height,
                               void *user) = NULL;

static PyObject *sbinfo_to_dict(void *ptr)
{
    const SuperBlockInfo *sb = (const SuperBlockInfo *)ptr;
    return Py_BuildValue(
        "{sI sI sI sI sI s:d}",
        "sb_org_x", sb->sb_org_x,
        "sb_org_y", sb->sb_org_y,
        "sb_width", sb->sb_width,
        "sb_height", sb->sb_height,
        "sb_qindex", sb->sb_qindex,
        "beta", sb->beta);
}

int *get_deltaq_offset_trampoline(SuperBlockInfo *sb_info_array, uint32_t sb_count,
                                int32_t picture_number, int32_t frame_type, void *user) {
    int *deltaq_map = NULL;
    Callback *cb = &g_callbacks[CB_GET_DELTAQ_OFFSET];
    if (cb->py_callable) {
        py_trampoline(cb->py_callable, cb->cb_fmt, &deltaq_map,
                      sb_info_array, sb_count, sbinfo_to_dict, /* LT: void*, length, transform func */
                      sb_count, /* u: uint32_t */
                      picture_number, /* i: int32_t */
                      frame_type); /* i: uint32_t */
    }
    return deltaq_map;
}

typedef struct {
    uint8_t  *buf;
    uint32_t  width, height;
    uint32_t  stride;
    uint32_t  org_x, org_y;
} BufferData;

/**
 * Create a Python list from a buffer.
 * The list will be a 2D list where each inner list represents a row of the buffer.
 */
static PyObject *buffer_to_list(void *ptr)
{
    BufferData *bd = (BufferData *)ptr;
    if (!bd->buf || bd->width <= 0 || bd->height <= 0 || bd->stride <= 0)
        return NULL;

    PyObject *list = PyList_New(bd->height);
    if (!list)
        return NULL;

    for (int i = 0; i < bd->height; ++i) {
        PyObject *row = PyList_New(bd->width);
        if (!row) {
            Py_DECREF(list);
            return NULL;
        }
        int buf_row_start = (bd->org_x + i) * bd->stride + bd->org_y;
        for (int j = 0; j < bd->width; ++j) {
            PyObject *value = PyLong_FromLong((long)bd->buf[buf_row_start + j]);
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

void recv_frame_feedback_trampoline(uint8_t *buffer_y, uint8_t *buffer_cb, uint8_t *buffer_cr, uint32_t picture_number,
                                    u_int32_t bytes_used, uint32_t origin_x, uint32_t origin_y, uint32_t stride_y,
                                    uint32_t stride_cb, uint32_t stride_cr, uint32_t width, uint32_t height,
                                    void *user) {
    Callback *cb = &g_callbacks[CB_RECV_FRAME_FEEDBACK];
    if (cb->py_callable) {
        BufferData bd_y = { buffer_y, width, height, stride_y, origin_x, origin_y };
        BufferData bd_cb = { buffer_cb, width / 2, height / 2, stride_cb, origin_x / 2, origin_y / 2 };
        BufferData bd_cr = { buffer_cr, width / 2, height / 2, stride_cr, origin_x / 2, origin_y / 2 };
        py_trampoline(cb->py_callable, cb->cb_fmt, NULL,
                      picture_number, /* u: uint32_t */
                      bytes_used, /* u: uint32_t */
                      width, /* u: uint32_t */
                      height, /* u: uint32_t */
                      &bd_y , buffer_to_list, /* T: void*, transform func */
                      &bd_cb, buffer_to_list, /* T: void*, transform func */
                      &bd_cr, buffer_to_list /* T: void*, transform func */
        );
    }
}

typedef struct {
    const uint8_t *data;
    uint32_t       size;
} BytesBuf;

static PyObject *bytes_from_buf(void *ptr)
{
    const BytesBuf *bb = (const BytesBuf *)ptr;
    return PyBytes_FromStringAndSize((const char *)bb->data, bb->size);
}

void recv_picture_feedback_trampoline(uint8_t *bitStream, uint32_t bitstream_size, uint32_t picture_number, void *user)
{
    Callback *cb = &g_callbacks[CB_RECV_PICTURE_FEEDBACK];
    if (cb->py_callable) {
        BytesBuf bb = { bitStream, bitstream_size };
        py_trampoline(cb->py_callable, "(Tu)v", NULL,
                      &bb, bytes_from_buf, /* T: void*, transform func */
                      picture_number); /* u: uint32_t */
    }
}


