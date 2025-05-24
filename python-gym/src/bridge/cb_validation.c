#include "cb_validation.h"
#include <regex.h>

static int count_fmt_args(const char *fmt) {
    regex_t regex;
    regmatch_t matches[2];

    if (regcomp(&regex, "^\\(([^()]*)\\).$", REG_EXTENDED) != 0) 
        return -1;

    int rc = regexec(&regex, fmt, 2, matches, 0);
    if (rc != 0) {
        regfree(&regex);
        return -1;
    }

    int count = matches[1].rm_eo - matches[1].rm_so;

    regfree(&regex);
    return count;
}

int validate_callable_signature(PyObject *callable, const char *fmt)
{
    if (!PyCallable_Check(callable)) {
        PyErr_SetString(PyExc_TypeError, "object must be callable");
        return -1;
    }

    PyObject *code = PyObject_GetAttrString(callable, "__code__");
    if (!code)
        return -1;

    if (!PyCode_Check(code)) {
        Py_DECREF(code);
        PyErr_SetString(PyExc_TypeError, "callable has no valid __code__");
        return -1;
    }

    PyCodeObject *co = (PyCodeObject *)code;
    const int expected = count_fmt_args(fmt);
    if (expected == -1) {
        Py_DECREF(code);
        PyErr_Format(PyExc_ValueError,
            "invalid format string: %s", fmt);
        return -1;
    }

    const int nargs = co->co_argcount + co->co_posonlyargcount;
    const int has_varargs = (co->co_flags & CO_VARARGS) != 0;
    const int has_varkw = (co->co_flags & CO_VARKEYWORDS) != 0;

    if (has_varargs || has_varkw) {
        Py_DECREF(code);
        PyErr_SetString(PyExc_TypeError,
            "callback may not use *args or **kwargs");
        return -1;
    }

    if (nargs != expected) {
        Py_DECREF(code);
        PyErr_Format(PyExc_TypeError,
            "callback must take exactly %d positional arguments (got %d)",
            expected, nargs);
        return -1;
    }

    Py_DECREF(code);
    return 0;
}