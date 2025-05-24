#ifndef PY_TRAMPOLINE_H
#define PY_TRAMPOLINE_H

#include <Python.h>

/* --------------------------------------------------------------------
 *  py_trampoline() – call a Python callable with typed C arguments
 *
 *  cb         : PyCallable to invoke
 *  fmt        : format string of the form "(args...)r"
 *               - args... : one char per positional argument
 *               - r       : return value type
 *
 *               Supported argument codes:
 *                    i  → signed int/long           → PyLong_FromLong
 *                    u  → unsigned int/long         → PyLong_FromUnsignedLong
 *                    d  → double                    → PyFloat_FromDouble
 *                    b  → int/bool                  → PyBool_FromLong
 *                    s  → const char*               → PyUnicode_FromString
 *                    O  → PyObject* (borrowed ref)  → INCREF + pass-through
 *                    M  → uint8_t*,int,int          → matrix → list of lists
 *                    L  → uint8_t*,int              → list of ints
 *
 *               Supported return codes:
 *                    i  → int                       ← PyLong_AsLong
 *                    u  → unsigned int              ← PyLong_AsUnsignedLong
 *                    d  → double                    ← PyFloat_AsDouble
 *                    b  → bool                      ← PyBool_Check / Py_True
 *                    O  → PyObject*                 ← returned with INCREF
 *                    v  → void                      ← return ignored
 *
 *  ret        : pointer to return value (output, may be NULL if return is 'v')
 *  …          : positional arguments matching the format string
 *
 *  Returns 0 on success, -1 on error (with Python exception set).
 * ------------------------------------------------------------------ */
int py_trampoline(PyObject *cb, const char *fmt, void* ret, ...);

#endif /* PY_TRAMPOLINE_H */