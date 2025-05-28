#include "py_trampoline.h"
#include "cb_validation.h"
#include "../../../Source/API/EbSvtAv1Enc.h"
#include <regex.h>
#include <stdbool.h>
#include <stdint.h>

typedef union {
    int8_t   i8;  uint8_t  u8;
    int16_t i16;  uint16_t u16;
    int32_t i32;  uint32_t u32;
    int64_t i64;  uint64_t u64;
    double   d; const void *p;
} ArgBuf;

static PyObject *PyConvert(char code, const void *addr)
{
    switch (code) {
        case 'c':  return PyLong_FromLong(*(const int8_t  *)addr);
        case 'C':  return PyLong_FromUnsignedLong(*(const uint8_t *)addr);
        case 'h':  return PyLong_FromLong(*(const int16_t *)addr);
        case 'H':  return PyLong_FromUnsignedLong(*(const uint16_t*)addr);
        case 'i':  return PyLong_FromLong(*(const int32_t *)addr);
        case 'u':  return PyLong_FromUnsignedLong(*(const uint32_t*)addr);
        case 'q':  return PyLong_FromLongLong(*(const int64_t *)addr);
        case 'Q':  return PyLong_FromUnsignedLongLong(*(const uint64_t*)addr);
        case 'd':  return PyFloat_FromDouble(*(const double*)addr);
        case 'b':  return PyBool_FromLong(*(const int32_t*)addr);
        case 's':  return PyUnicode_FromString(*(const char *const *)addr);
        case 'O': {
            PyObject *o = *(PyObject *const *)addr;
            Py_XINCREF(o);
            return o;
        }
        default:
            PyErr_Format(PyExc_ValueError, "unknown element code '%c'", code);
            return NULL;
    }
}

static inline size_t elem_size(char code)
{
    switch (code) {
        case 'c': case 'C': return sizeof(uint8_t);
        case 'h': case 'H': return sizeof(uint16_t);
        case 'i': case 'u': case 'b': return sizeof(uint32_t);
        case 'q': case 'Q': case 'd': return sizeof(uint64_t);
        case 'T': case 'O': case 's': return sizeof(void *);
        default: return sizeof(int);
    }
}

static PyObject *PyList_Create(char code, const void *buf, int len, PyObject *(*transform)(void *))
{
    if (!buf || len < 0) {
        PyErr_SetString(PyExc_ValueError, "invalid buffer/length for list");
        return NULL;
    }

    PyObject *list = PyList_New(len);
    if (!list) return NULL;

    size_t esz = elem_size(code);
    const char *base = (const char *)buf;

    for (int i = 0; i < len; ++i) {
        PyObject *item = NULL;

        if (code == 'T') {
            void *ptr = ((void *const *)buf)[i];
            item = transform(ptr);
            if (!item) {
                Py_DECREF(list);
                return NULL;
            }
            if (item == Py_None) Py_INCREF(Py_None);
        } 
        else {
            const void *addr = base + (size_t)i * esz;
            item = PyConvert(code, addr);
        }

        if (!item || PyList_SetItem(list, i, item) < 0) {
            Py_DECREF(list);
            return NULL;
        }
    }
    return list;
}

static PyObject *PyMat_Create(char code, const void *buf, int w, int h, PyObject *(*transform)(void *))
{
    if (!buf || w <= 0 || h <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid buffer/dimensions for matrix");
        return NULL;
    }

    PyObject *mat = PyList_New(h);
    if (!mat) return NULL;

    size_t esz = elem_size(code);
    const char *base = (const char *)buf;

    for (int r = 0; r < h; ++r) {
        const void *row_start = base + (size_t)r * w * esz;
        PyObject *row = PyList_Create(code, row_start, w, transform);
        if (!row || PyList_SetItem(mat, r, row) < 0) {
            Py_DECREF(mat);
            return NULL;
        }
    }
    return mat;
}

static int
PyReadArgs(const char *fmt, PyObject *tuple, va_list ap, char *retcode_out)
{
    int pos = 0;
    for (const char *p = fmt + 1; *p && *p != ')'; ++p) {
        char code = *p;
        PyObject *arg = NULL;

        if (code == 'L' || code == 'M') {                /* containers */
            char sub = *++p;
            if (!sub || sub == ')') {
                PyErr_SetString(PyExc_ValueError, "missing sub‑code for L/M");
                return -1;
            }

            if (code == 'L') {
                const void *buf = va_arg(ap, const void *);
                int len          = va_arg(ap, int);

                PyObject *(*tx)(void *) = NULL;
                if (sub == 'T') {
                    tx = va_arg(ap, PyObject *(*)(void *));
                    if (!tx) {
                        PyErr_SetString(PyExc_ValueError, "missing transform argument");
                        return -1;
                    }
                }

                arg = PyList_Create(sub, buf, len, tx);

            } else {                     /* 'M' */
                const void *buf = va_arg(ap, const void *);
                int w = va_arg(ap, int), h = va_arg(ap, int);

                PyObject *(*tx)(void *) = NULL;
                if (sub == 'T') {
                    tx = va_arg(ap, PyObject *(*)(void *));
                    if (!tx) {
                        PyErr_SetString(PyExc_ValueError, "missing transform argument");
                        return -1;
                    }
                }

                arg = PyMat_Create(sub, buf, w, h, tx);
            }
        } else { 
            ArgBuf buf;
            switch (code) {
                case 'b': case 'c': case 'C': case 'h': 
                case 'H': case 'i': case 'u':
                    buf.i32 = va_arg(ap, int);
                    break;
                case 'q':
                    buf.i64 = va_arg(ap, long long); 
                    break;
                case 'Q':
                    buf.u64 = va_arg(ap, unsigned long long);     
                    break;
                case 'd':
                    buf.d = va_arg(ap, double);
                    break;
                case 's':
                    buf.p = va_arg(ap, const char *);
                    break;
                case 'O': case 'B':
                    buf.p = va_arg(ap, void *);
                    break;
                case 'T': {
                    void *data = va_arg(ap, void *);
                    PyObject *(*transform)(void *) = va_arg(ap, PyObject *(*)(void *));
                    if (!transform) {
                        PyErr_SetString(PyExc_ValueError, "missing transform argument");
                        return -1;
                    }

                    arg = transform(data);
                    break;
                }
                default:
                    PyErr_Format(PyExc_ValueError, "unknown format code '%c'", code);
                    return -1;
            }
            arg = PyConvert(code, &buf);
        }

        if (!arg || PyTuple_SetItem(tuple, pos++, arg) < 0)
            return -1;
    }

    *retcode_out = fmt[strlen(fmt) - 1];
    return 0;
}

int py_trampoline(PyObject *cb, const char *fmt, void* ret, ...)
{
    PyGILState_STATE g = PyGILState_Ensure();

    if (!validate_callable_signature(cb, fmt)) {
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
    char retfmt;
    if (PyReadArgs(fmt, tuple, args, &retfmt) != 0) {
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
        case 'L': {
            if (!PyList_Check(result)) {
                PyErr_SetString(PyExc_TypeError, "Expected a Python list");
                rc = -1;
                break;
            }

            Py_ssize_t n = PyList_Size(result);
            if (n < 0) { 
                rc = -1; 
                break; 
            }

            int *out = (int*)ret;
            out = (int *)malloc((size_t)n * sizeof(int));
            if (!out) {
                PyErr_NoMemory();
                rc = -1;
                break;
            }

            for (Py_ssize_t i = 0; i < n; ++i) {
                PyObject *item = PyList_GET_ITEM(result, i);   /* borrowed ref */
                if (!PyLong_Check(item)) {
                    PyErr_Format(PyExc_TypeError, "List element %zd is not an int", i);
                    free(out);
                    rc = -1;
                    break;
                }
                out[i] = (int)PyLong_AsLong(item);
            }
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
