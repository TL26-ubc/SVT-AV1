#include "tl26_flags.h"

void initialize_python() {
    // Initialize the Python interpreter
    Py_Initialize();
    PyObject *pName = PyUnicode_DecodeFSDefault("tl26.utils");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    PyObject *python_greetings = PyObject_GetAttrString(pModule, "hello_SVTAV1");
    if (PyCallable_Check(python_greetings)) {
        PyObject *pValue = PyObject_CallObject(python_greetings, NULL);
        if (pValue != NULL) {
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
        }
    } else {
        PyErr_Print();
    }
    Py_XDECREF(python_greetings);
    Py_DECREF(pModule);
}

void finalize_python() {
    // Finalize the Python interpreter
    Py_Finalize();
}
