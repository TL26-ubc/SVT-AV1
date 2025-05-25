#ifndef CB_VALIDATION_H
#define CB_VALIDATION_H

#include <Python.h>

int validate_callable_signature(PyObject *callable, const char *fmt);

#endif /* CB_VALIDATION_H */