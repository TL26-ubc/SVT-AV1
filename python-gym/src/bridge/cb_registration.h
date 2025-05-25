#ifndef CB_REGISTRATION_H
#define CB_REGISTRATION_H

#include <Python.h>

typedef enum CallbackEnum { 
    CB_GET_DELTAQ_OFFSET,
    CB_RECV_FRAME_FEEDBACK, 
    CB_ENUM_COUNT
} CallbackEnum;

typedef struct {
    PyObject *py_callable;
    void *c_trampoline;
    char *cb_fmt;
} Callback;

extern Callback g_callbacks[CB_ENUM_COUNT];

int pybridge_set_cb(CallbackEnum cb, PyObject *callable);

#endif /* CB_REGISTRATION_H */