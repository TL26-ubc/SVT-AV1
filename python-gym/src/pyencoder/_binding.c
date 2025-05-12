#define main app_main
#include "../Source/App/app_main.c"
#undef main

#include <Python.h>
#include <pybridge.h>

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
    PyObject *py_sb = Py_None;
    if (!PyArg_ParseTuple(args, "O", &py_sb))
        return PyErr_Format(PyExc_TypeError,
                            "unable to parse get_deltaq_offset");

    if (py_sb != Py_None && !PyCallable_Check(py_sb))
        return PyErr_Format(PyExc_TypeError,
                            "get_deltaq_offset must be callable or None");

    // keep global ref
    pybridge_set_cb(py_sb == Py_None ? NULL : py_sb);

    static PluginCallbacks cbs;
    cbs.user_get_deltaq_offset = get_deltaq_offset_cb;

    if (svt_av1_enc_set_callbacks(&cbs) != EB_ErrorNone)
        return PyErr_Format(PyExc_TypeError,
            "failed to set callbacks");

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
