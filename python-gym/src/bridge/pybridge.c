#define PY_SSIZE_T_CLEAN
#include "pybridge.h"

static PyObject *py_get_deltaq_offset_obj = NULL;
int (*get_deltaq_offset_cb)(
    unsigned,
    unsigned,
    unsigned,
    uint8_t,
    uint16_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    uint8_t,
    double,
    bool,
    void*) = NULL;

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
    uint8_t encoder_bit_depth,
    double beta,
    bool is_intra, 
    void *user
) {
    int deltaq = 0; // Default
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Call the python function
    PyObject *ret = NULL;
    if (py_get_deltaq_offset_obj && PyCallable_Check(py_get_deltaq_offset_obj)) {
        PyObject *args = Py_BuildValue(
            "(IIIBHiiiiiiiiBdO)",
            sb_index, sb_org_x, sb_org_y,
            sb_qindex, sb_final_blk_cnt,
            mi_row_start, mi_row_end,
            mi_col_start, mi_col_end,
            tg_horz_boundary, tile_row, tile_col,
            tile_rs_index, encoder_bit_depth,
            beta, PyBool_FromLong(is_intra));
        if (!args) { // Py_BuildValue failed
            PyErr_Print();
            goto done;
        }

        ret = PyObject_CallObject(py_get_deltaq_offset_obj, args);
        Py_DECREF(args);

        if (!ret) { // callback raised error
            PyErr_Print();
            goto done;
        }

        if (PyLong_Check(ret)) {
            deltaq = (int)PyLong_AsLong(ret);
        } else {
            PyErr_Format(PyExc_TypeError,
                         "QP callback must return int, got %.200s",
                         Py_TYPE(ret)->tp_name);
            PyErr_Print();
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "get_deltaq_offset callback not set or not callable");
        PyErr_Print();
    }

    done:
        Py_XDECREF(ret);
        PyGILState_Release(gstate);
        return deltaq;
}

void pybridge_set_cb(PyObject *callable)
{
    Py_XINCREF(callable);
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = callable;

    // Update the global pointer for the encoder
    get_deltaq_offset_cb = callable ? get_deltaq_offset_trampoline : NULL;
}

void pybridge_clear(void)
{
    Py_XDECREF(py_get_deltaq_offset_obj);
    py_get_deltaq_offset_obj = NULL;
    get_deltaq_offset_cb = NULL;
}
