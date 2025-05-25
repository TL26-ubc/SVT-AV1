#include "cb_registration.h"
#include <stdbool.h>
#include "pybridge.h"
#include "cb_validation.h"

Callback g_callbacks[CB_ENUM_COUNT] = {
    /* CB_GET_DELTAQ_OFFSET */   { NULL, NULL, "(IIIIIiiiiiiiidMMMIIIidb)i" },
    /* CB_RECV_FRAME_FEEDBACK */ { NULL, NULL, "(iiiOOO)v" },
};

static int set_cb_ptr(CallbackEnum cb, bool unset) {
    switch (cb) {
        case CB_GET_DELTAQ_OFFSET:
            get_deltaq_offset_cb = unset ? NULL : get_deltaq_offset_trampoline;
            return 0;
        case CB_RECV_FRAME_FEEDBACK:
            recv_frame_feedback_cb = unset ? NULL : recv_frame_feedback_trampoline;
            return 0;
        default:
            return -1;
    }
}

static int pybridge_clear(CallbackEnum cb)
{
    Callback *cb_struct = &g_callbacks[cb];
    Py_XDECREF(cb_struct->py_callable);
    cb_struct->py_callable = NULL;
    
    // Remove the global pointer for the encoder
    return set_cb_ptr(cb, false);
}

int pybridge_set_cb(CallbackEnum cb, PyObject *callable)
{
    // Unset if none
    if (callable == Py_None) {
        return pybridge_clear(cb);
    }

    Callback *cb_struct = &g_callbacks[cb];

    if (validate_callable_signature(callable, cb_struct->cb_fmt) != 0)
        return -1;

    Py_XINCREF(callable);
    Py_XDECREF(cb_struct->py_callable);
    cb_struct->py_callable = callable;

    // Update the global pointer for the encoder
    return set_cb_ptr(cb, false);
}
