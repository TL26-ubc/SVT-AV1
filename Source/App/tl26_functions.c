#include "tl26_flags.h"
#include "svt_log.h"

void initialize_python() {
    Py_Initialize();

    // Convert module name to a Python object
    PyObject *pName = PyUnicode_DecodeFSDefault("tl26.utils");
    if (!pName) {
        PyErr_Print();
        panic("Error: Failed to decode module name 'tl26.utils'.\n");
    }

    // Import the module
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (!pModule) {
        PyErr_Print();
        panic(stderr, "Error: Failed to import module 'tl26.utils'.\n");
    }

    SVT_INFO("Successfully loaded tl26 module!");
}

void finalize_python() {
    // Finalize the Python interpreter
    Py_Finalize();
}
