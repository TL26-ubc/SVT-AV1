#include "tl26_flags.h"

#ifdef TL26_RL
static PyObject *f_frame_report_feedback, *f_sb_send_offset_request;

static void initialize_communications() {
    // import the functions from tl.feedback
    PyObject *pFeedbackName   = PyUnicode_DecodeFSDefault("tl26.feedback");
    PyObject *pFeedbackModule = PyImport_Import(pFeedbackName);
    Py_DECREF(pFeedbackName);
    f_frame_report_feedback = PyObject_GetAttrString(pFeedbackModule, "frame_report_feedback");
    Py_DECREF(pFeedbackModule);

    // import the functions from tl.request
    PyObject *pRequestName   = PyUnicode_DecodeFSDefault("tl26.request");
    PyObject *pRequestModule = PyImport_Import(pRequestName);
    Py_DECREF(pRequestName);
    f_sb_send_offset_request = PyObject_GetAttrString(pRequestModule, "sb_send_offset_request");
    Py_DECREF(pRequestModule);
}

void initialize_python() {
    // Initialize the Python interpreter
    Py_Initialize();

    PyGILState_STATE state = PyGILState_Ensure();

    PyObject *pName   = PyUnicode_DecodeFSDefault("tl26.utils");
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

    // Initialize the communication functions for SVT-AV1 and Python
    initialize_communications();

    PyGILState_Release(state);
}

void finalize_python() {
    // Free the function objects
    Py_XDECREF(f_frame_report_feedback);
    Py_XDECREF(f_sb_send_offset_request);

    // Finalize the Python interpreter
    Py_Finalize();
}

PyObject *get_f_frame_report_feedback() {
    return f_frame_report_feedback;
}

PyObject *get_f_sb_send_offset_request() {
    return f_sb_send_offset_request;
}

#endif
