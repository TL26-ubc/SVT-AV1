#include "tl26_flags.h"
#include <stdlib.h>

void initialize_python() {
    Py_Initialize();

    // Convert module name to a Python object
    PyObject *pName = PyUnicode_DecodeFSDefault("tl26.utils");
    if (!pName) {
        PyErr_Print();
        fprintf(stderr, "Error: Failed to decode module name 'tl26.utils'.\n");
        abort();
    }

    // Import the module
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    if (!pModule) {
        PyErr_Print();
        fprintf(stderr, "Error: Failed to import module 'tl26.utils'.\n");
        abort();
    }

    printf("Successfully loaded tl26 module!\n");
}

void finalize_python() {
    // Finalize the Python interpreter
    Py_Finalize();
}
