#define PY_SSIZE_T_CLEAN
#include "pybridge.h"

static PyObject *py_sb_qp_obj = NULL;
int (*g_py_sb_qp_cb)(int, int, uint64_t, uint64_t) = NULL;

static int sb_qp_trampoline(int row, int col,
                            uint64_t stat1, uint64_t stat2)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    // Call the python function
    PyObject *ret = PyObject_CallFunction(
        py_sb_qp_obj, "(iiKK)", row, col, stat1, stat2);

    if (!ret || !PyLong_Check(ret))
        return Eb

    qp = (int)PyLong_AsLong(ret);

    Py_XDECREF(ret);
    PyGILState_Release(gstate);
    return qp;
}

void pybridge_set_cb(PyObject *callable)
{
    Py_XINCREF(callable);
    Py_XDECREF(py_sb_qp_obj);
    py_sb_qp_obj = callable;

    // Update the global pointer for the encoder
    g_py_sb_qp_cb = callable ? sb_qp_trampoline : NULL;
}

void pybridge_clear(void)
{
    Py_XDECREF(py_sb_qp_obj);
    py_sb_qp_obj = NULL;
    g_py_sb_qp_cb = NULL;
}
