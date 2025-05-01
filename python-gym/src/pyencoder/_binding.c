#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include "pybridge.h"

// from app_entry.c
extern int svt_av1_app_entry(int argc, char *argv[]);

// Register the callback
static int
register_qp_cb(PyObject *maybe_callable)
{
    if (maybe_callable && maybe_callable != Py_None) {
        if (!PyCallable_Check(maybe_callable)) {
            PyErr_SetString(PyExc_TypeError,
                            "second argument must be callable or None");
            return -1;
        }
        pybridge_set_cb(maybe_callable);
    } else {
        pybridge_clear();
    }
    return 0;
}

/*  Python: run(argv: list[str], qp_cb: Optional[callable]) */
static PyObject *
py_run(PyObject *self, PyObject *args)
{
    PyObject *py_argv, *py_cb = Py_None;
    if (!PyArg_ParseTuple(args, "O!|O", &PyList_Type, &py_argv, &py_cb))
        return NULL;

    if (register_qp_cb(py_cb) < 0)
        return NULL;

    // Build C argc/argv array
    Py_ssize_t argc = PyList_GET_SIZE(py_argv);
    char **argv = calloc((size_t)argc + 1, sizeof(char*));
    if (!argv)
        return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < argc; ++i) {
        PyObject *utf8 = PyUnicode_AsUTF8String(PyList_GET_ITEM(py_argv, i));
        if (!utf8) {
            free(argv);
            return NULL; // propagate Unicode error
        }
        argv[i] = PyBytes_AsString(utf8);
        PyList_SET_ITEM(py_argv, i, utf8);
    }

    // Start the encoder app
    int ret;
    Py_BEGIN_ALLOW_THREADS
        ret = svt_av1_app_entry((int)argc, argv);
    Py_END_ALLOW_THREADS

    free(argv);

    if (ret != 0)
        return PyErr_Format(PyExc_RuntimeError,
                            "encoder returned non-zero status %d", rc);

    Py_RETURN_NONE;
}

/*  Module boiler-plate */
static PyMethodDef mod_methods[] = {
    {"run", (PyCFunction)py_run, METH_VARARGS,
     "run(argv:list[str], qp_callback:callable|None) -> None"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,
    "pyencoder._binding",
    "Python ↔︎ SVT-AV1 bridge",
    -1,
    mod_methods
};

PyMODINIT_FUNC
PyInit__binding(void)
{
    return PyModule_Create(&moddef);
}
