#define main app_main
#include "../Source/App/app_main.c"
#undef main

#ifndef SVT_ENABLE_USER_CALLBACKS
#define SVT_ENABLE_USER_CALLBACKS 1
#endif

#include <Python.h>
#include "../bridge/cb_validation.h"
#include "../bridge/cb_registration.h"
#include "../bridge/pybridge.h"
#include "../Source/Lib/Globals/enc_callbacks.h"
#include "../Source/API/EbSvtAv1Enc.h"
#include <EbSvtAv1Enc.h>

static PyObject *
py_run_app(PyObject *self, PyObject *args)
{
    PyObject *py_argv;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_argv))
        return NULL;

    int argc = (int)PyList_Size(py_argv);
    char **argv = malloc((argc + 1) * sizeof(char*));
    for (int i = 0; i < argc; i++) {
        PyObject *item = PyList_GetItem(py_argv, i);
        argv[i] = strdup(PyUnicode_AsUTF8(item));
    }
    argv[argc] = NULL;

    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = app_main(argc, argv);
    Py_END_ALLOW_THREADS

    for (int i = 0; i < argc; i++) free(argv[i]);
    free(argv);

    if (rc != 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "SvtAv1EncApp returned non-zero exit code %d", rc);
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
py_register_cbs(PyObject *self, PyObject *args)
{
    PyObject *py_get_deltaq_offset = Py_None;
    PyObject *py_frame_feedback = Py_None;

    if (!PyArg_ParseTuple(args, "|OO", &py_get_deltaq_offset, &py_frame_feedback))
        return PyErr_Format(PyExc_TypeError, "unable to parse callback arguments");

    if (py_get_deltaq_offset != Py_None && !PyCallable_Check(py_get_deltaq_offset))
        return PyErr_Format(PyExc_TypeError, "get_deltaq_offset must be callable or None");
    
    if (py_frame_feedback != Py_None && !PyCallable_Check(py_frame_feedback))
        return PyErr_Format(PyExc_TypeError, "frame_feedback must be callable or None");

    pybridge_set_cb(CB_GET_DELTAQ_OFFSET, py_get_deltaq_offset);
    pybridge_set_cb(CB_RECV_FRAME_FEEDBACK, py_frame_feedback);
    // Set the callbacks in the SVT-AV1 encoder

    static PluginCallbacks cbs;
    cbs.user_get_deltaq_offset = get_deltaq_offset_cb;
    cbs.user_frame_feedback = recv_frame_feedback_cb;

    if (svt_av1_enc_set_callbacks(&cbs) != EB_ErrorNone)
        return PyErr_Format(PyExc_RuntimeError, "failed to set callbacks");

    Py_RETURN_NONE;
}

static PyMethodDef SvtAppMethods[] = {
    {"run", py_run_app, METH_VARARGS, "Run the SVT-AV1 encoder CLI in-process."},
    {"register_callbacks", py_register_cbs, METH_VARARGS, "Attach callbacks to the SVT-AV1 encoder."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef svtappmodule = {
    PyModuleDef_HEAD_INIT,
    "_svtapp",   /* name of module */
    NULL,        /* module doc */
    -1,
    SvtAppMethods
};

PyMODINIT_FUNC
PyInit__svtapp(void)
{
    return PyModule_Create(&svtappmodule);
}
