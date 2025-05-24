#define PY_SSIZE_T_CLEAN
#include "pybridge.h"

static PyObject *py_get_deltaq_offset_obj = NULL;
static PyObject *py_frame_feedback_obj = NULL;
static PyObject *py_sb_feedback_obj = NULL;

static PyObject* create_python_list_from_buffer(uint8_t* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer or dimensions");
        return NULL;
    }

    PyObject* list = PyList_New(height);
    if (!list)
        return NULL;

    for (int i = 0; i < height; i++) {
        PyObject* row = PyList_New(width);
        if (!row) {
            Py_DECREF(list);
            return NULL;
        }

        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            PyObject* value = PyLong_FromLong((long)buffer[index]);
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

int (*get_deltaq_offset_cb)(
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
    void* user);

void (*frame_feedback_cb)(
    int,
    int,
    int,
    int,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    int,
    void*) = NULL;

void (*sb_feedback_cb)(
    int,
    unsigned,
    unsigned,
    unsigned,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    uint8_t*,
    uint8_t*,
    uint8_t*,
    uint16_t,
    uint16_t,
    void*) = NULL;

// QP offset callback trampoline
static int get_deltaq_offset_trampoline(
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
    int deltaq = 0; // Default
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (py_get_deltaq_offset_obj && PyCallable_Check(py_get_deltaq_offset_obj)) {
        
        PyObject *py_buffer_y = create_python_list_from_buffer(buffer_y, sb_width, sb_height);
        PyObject *py_buffer_cb = create_python_list_from_buffer(buffer_cb, sb_width / 2, sb_height / 2);
        PyObject *py_buffer_cr = create_python_list_from_buffer(buffer_cr, sb_width / 2, sb_height / 2);

        if (py_buffer_y && py_buffer_cb && py_buffer_cr) {

            PyObject *args = Py_BuildValue(
                "(IIIBHiiiiiiiiiOOOHHBidO)",
                sb_index,                    // I: unsigned
                sb_org_x,                    // I: unsigned
                sb_org_y,                    // I: unsigned
                sb_qindex,                   // B: uint8_t
                sb_final_blk_cnt,            // H: uint16_t
                mi_row_start,                // i: int32_t
                mi_row_end,                  // i: int32_t
                mi_col_start,                // i: int32_t
                mi_col_end,                  // i: int32_t
                tg_horz_boundary,            // i: int32_t
                tile_row,                    // i: int32_t
                tile_col,                    // i: int32_t
                tile_rs_index,               // i: int32_t
                picture_number,              // i: int32_t 
                py_buffer_y,                 // O: PyObject* 
                py_buffer_cb,                // O: PyObject* 
                py_buffer_cr,                // O: PyObject* 
                sb_width,                    // H: uint16_t 
                sb_height,                   // H: uint16_t 
                encoder_bit_depth,           // B: uint8_t
                qindex,                      // i: int32_t 
                beta,                        // d: double
                PyBool_FromLong(type == 1)   // O: bool (is_intra) 
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
                PyErr_Format(PyExc_TypeError,
                             "QP callback must return int, got %.200s",
                             Py_TYPE(ret)->tp_name);
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

static void frame_feedback_trampoline(
    int picture_number,
    int temporal_layer_index,
    int qp,
    int avg_qp,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    int picture_stream_size,
    void *user
) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (py_frame_feedback_obj && PyCallable_Check(py_frame_feedback_obj)) {
        PyObject *args = Py_BuildValue(
            "(iiiidddddddddi)",
            picture_number,
            temporal_layer_index,
            qp,
            avg_qp,
            luma_psnr,
            cb_psnr,
            cr_psnr,
            mse_y,
            mse_u,
            mse_v,
            luma_ssim,
            cb_ssim,
            cr_ssim,
            picture_stream_size);
        
        if (args) {
            PyObject *ret = PyObject_CallObject(py_frame_feedback_obj, args);
            Py_XDECREF(ret);
            Py_DECREF(args);
        }
        
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    }

    PyGILState_Release(gstate);
}


static void sb_feedback_trampoline(
    int picture_number,
    unsigned sb_index,
    unsigned sb_origin_x,
    unsigned sb_origin_y,
    double luma_psnr,
    double cb_psnr,
    double cr_psnr,
    double mse_y,
    double mse_u,
    double mse_v,
    double luma_ssim,
    double cb_ssim,
    double cr_ssim,
    uint8_t *buffer_y,
    uint8_t *buffer_cb,
    uint8_t *buffer_cr,
    uint16_t sb_width,
    uint16_t sb_height,
    void *user
) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    if (py_sb_feedback_obj && PyCallable_Check(py_sb_feedback_obj)) {
        
        PyObject *py_y = create_python_list_from_buffer(buffer_y, sb_width, sb_height);
        PyObject *py_cb = create_python_list_from_buffer(buffer_cb, sb_width / 2, sb_height / 2);
        PyObject *py_cr = create_python_list_from_buffer(buffer_cr, sb_width / 2, sb_height / 2);

        if (py_y && py_cb && py_cr) {
            PyObject *args = Py_BuildValue(
                "iIIIdddddddddOOO",
                picture_number,
                sb_index,
                sb_origin_x,
                sb_origin_y,
                luma_psnr,
                cb_psnr,
                cr_psnr,
                mse_y,
                mse_u,
                mse_v,
                luma_ssim,
                cb_ssim,
                cr_ssim,
                py_y,
                py_cb,
                py_cr);

            if (args) {
                PyObject *ret = PyObject_CallObject(py_sb_feedback_obj, args);
                Py_XDECREF(ret);
                Py_DECREF(args);
            }
        }

        Py_XDECREF(py_y);
        Py_XDECREF(py_cb);
        Py_XDECREF(py_cr);
        
        if (PyErr_Occurred()) {
            PyErr_Print();
        }
    }

    PyGILState_Release(gstate);
}

void pybridge_set_callbacks(PyObject *get_deltaq_offset, PyObject *frame_feedback, PyObject *sb_feedback)
{
    // Set QP offset callback
    Py_XINCREF(get_deltaq_offset);
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = get_deltaq_offset;
    get_deltaq_offset_cb = get_deltaq_offset ? get_deltaq_offset_trampoline : NULL;

    // Set frame feedback callback
    Py_XINCREF(frame_feedback);
    Py_XDECREF(py_frame_feedback_obj);
    py_frame_feedback_obj = frame_feedback;
    frame_feedback_cb = frame_feedback ? frame_feedback_trampoline : NULL;

    // Set superblock feedback callback
    Py_XINCREF(sb_feedback);
    Py_XDECREF(py_sb_feedback_obj);
    py_sb_feedback_obj = sb_feedback;
    sb_feedback_cb = sb_feedback ? sb_feedback_trampoline : NULL;
}

void pybridge_clear(void)
{
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = NULL;
    get_deltaq_offset_cb = NULL;

    Py_XDECREF(py_frame_feedback_obj);
    py_frame_feedback_obj = NULL;
    frame_feedback_cb = NULL;

    Py_XDECREF(py_sb_feedback_obj);
    py_sb_feedback_obj = NULL;
    sb_feedback_cb = NULL;
}