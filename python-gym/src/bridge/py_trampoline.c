#include "py_trampoline.h"
#include <regex.h>
#include <stdbool.h>

static PyObject* PyList_Create(uint8_t* buffer, int length) {
    if (!buffer || length <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer or length");
        return NULL;
    }

    PyObject* list = PyList_New(length);
    if (!list) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate Python list");
        return NULL;
    }

    for (int i = 0; i < length; i++) {
        PyObject* value = PyLong_FromLong((long)buffer[i]);
        if (!value) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SetItem(list, i, value);  // steals reference
    }

    return list;
}

static PyObject* PyMat_Create(uint8_t* buffer, int width, int height) {
    if (!buffer || width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid buffer or dimensions");
        return NULL;
    }

    PyObject* mat = PyList_New(height);
    if (!mat) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate Python list");
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        int offset = i * width;
        PyObject* row = PyList_Create(&buffer[offset], width);
        if (!row) {
            Py_DECREF(mat);
            return NULL;
        }

        PyList_SetItem(mat, i, row);  // steals reference
    }

    return mat;
}

static bool validateArgFormat(const char *fmt) {
    regex_t r;
    int err = regcomp(&r, "^\\([^()]+\\).$", REG_EXTENDED);
    if (err) { return false; }
    err = regexec(&r, fmt, 0, NULL, 0);
    regfree(&r);
    return !err;
}

static int PyReadArgs(const char *fmt, PyObject *tuple, Py_ssize_t n_args, va_list args, char **pyfmt_ret, char *retfmt_ret) {
    if (n_args >= 64) {
        PyErr_SetString(PyExc_ValueError, "too many arguments in trampoline format (max 64)");
        return -1;
    }

    char pyfmt_buf[64] = {0};
    Py_ssize_t i;
    for (i = 1; i < n_args + 1; ++i) {
        PyObject *item = NULL;

        switch (fmt[i]) {
            case 'i': {
                long v = va_arg(args, long);
                item = PyLong_FromLong(v);
                strncat(pyfmt_buf, "i", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                break;
            }
            case 'u': {
                unsigned long v = va_arg(args, unsigned long);
                item = PyLong_FromUnsignedLong(v);
                strncat(pyfmt_buf, "I", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                break;
            }
            case 'd': {
                double v = va_arg(args, double);
                item = PyFloat_FromDouble(v);
                strncat(pyfmt_buf, "d", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                break;
            }
            case 'b': {
                int v = va_arg(args, int);
                item = PyBool_FromLong(v);
                strncat(pyfmt_buf, "O", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                break;
            }
            case 's': {
                const char *v = va_arg(args, const char *);
                item = PyUnicode_FromString(v);
                strncat(pyfmt_buf, "s", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                break;
            }
            case 'O': {
                PyObject *v = va_arg(args, PyObject *);
                if (v) {
                    Py_INCREF(v);
                    item = v;
                    strncat(pyfmt_buf, "O", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                }
                break;
            }
            case 'M': {
                uint8_t *b = va_arg(args, uint8_t *);
                int w = va_arg(args, int);
                int h = va_arg(args, int);

                PyObject *mat = PyMat_Create(b, w, h);
                if (mat) {
                    item = mat;
                    strncat(pyfmt_buf, "O", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                }
                break;
            }
            case 'L': {
                uint8_t *b = va_arg(args, uint8_t *);
                int l = va_arg(args, int);

                PyObject *list = PyList_Create(b, l);
                if (list) {
                    item = list;
                    strncat(pyfmt_buf, "O", sizeof(pyfmt_buf) - strlen(pyfmt_buf) - 1);
                }
                break;
            }
            case ')':
                PyErr_Format(PyExc_ValueError,
                            "format string terminated with %d args left", n_args - i);
                return -1;
            default:
                PyErr_Format(PyExc_ValueError,
                            "unknown format code '%c' at index %zd", fmt[i], i);
                return -1;
        }

        if (!item || PyTuple_SetItem(tuple, i - 1, item) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to allocate or set Python object");
            Py_XDECREF(item);
            return -1;
        }
    }

    if (fmt[i] != ')') {
        PyErr_SetString(PyExc_ValueError,
                "arg list terminated without terminating the format string");
        return -1;
    }

    *pyfmt_ret = strdup(pyfmt_buf);
    *retfmt_ret = fmt[i + 1];
    return 0;
}

int py_trampoline(PyObject *cb, const char *fmt, void* ret, ...)
{
    PyGILState_STATE g = PyGILState_Ensure();

    if (!PyCallable_Check(cb)) {
        PyErr_SetString(PyExc_TypeError, "callback object is not callable");
        PyGILState_Release(g);
        return -1;
    }

    // Validate format
    if (!validateArgFormat(fmt)) {
        PyErr_SetString(PyExc_RuntimeError, "Internal Error: Invalid trampoline format string");
        PyGILState_Release(g);
        return -1;
    }

    Py_ssize_t n_args = (Py_ssize_t)strlen(fmt) - 3;
    PyObject *tuple = PyTuple_New(n_args);
    if (!tuple) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate Python tuple");
        PyGILState_Release(g);
        return -1;
    }

    // Build args
    va_list args;
    va_start(args, ret);
    char *pyfmt;
    char retfmt;
    if (PyReadArgs(fmt, tuple, n_args, args, &pyfmt, &retfmt) != 0) {
        va_end(args);
        Py_DECREF(tuple);
        PyGILState_Release(g);
        return -1;
    }

    PyObject *result = PyObject_CallObject(cb, tuple);
    va_end(args);
    Py_DECREF(tuple);

    if (!result) {
        PyGILState_Release(g);
        return -1;
    }

    int rc = 0;
    switch (retfmt) {
        case 'i': {
            if (!PyLong_Check(result)) {
                PyErr_SetString(PyExc_TypeError, "Expected int return");
                rc = -1;
                break;
            }
            *(int*)ret = (int)PyLong_AsLong(result);
            break;
        }
        case 'u': {
            if (!PyLong_Check(result)) {
                PyErr_SetString(PyExc_TypeError, "Expected unsigned int return");
                rc = -1;
                break;
            }
            *(unsigned int*)ret = (unsigned int)PyLong_AsUnsignedLong(result);
            break;
        }
        case 'd': {
            if (!PyFloat_Check(result)) {
                PyErr_SetString(PyExc_TypeError, "Expected float return");
                rc = -1;
                break;
            }
            *(double*)ret = PyFloat_AsDouble(result);
            break;
        }
        case 'b': {
            if (!PyBool_Check(result)) {
                PyErr_SetString(PyExc_TypeError, "Expected bool return");
                rc = -1;
                break;
            }
            *(bool*)ret = (result == Py_True);
            break;
        }
        case 'O': {
            *(PyObject**)ret = result;  // transfer ownership to caller
            Py_INCREF(result);
            break;
        }
        case 'v': {
            // void return: do nothing
            break;
        }
        default:
            PyErr_Format(PyExc_ValueError, "Unknown return format code '%c'", retfmt);
            rc = -1;
    }

    Py_DECREF(result);
    PyGILState_Release(g);
    return rc;
}
