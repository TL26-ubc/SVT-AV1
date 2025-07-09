#pragma once
#include <pybind11/pybind11.h>

namespace pybridge {

enum class CallbackEnum {
    GetDeltaQOffset,
    RecvPictureFeedback,
    RecvPostEncodeStats,
    Count
};

struct Callback {
    pybind11::object py_func;
    void            *c_trampoline;
    int              n_args;
};

extern Callback* g_callbacks[static_cast<int>(CallbackEnum::Count)];

int pybridge_set_cb(CallbackEnum which, pybind11::object callable);

} // namespace pybridge

