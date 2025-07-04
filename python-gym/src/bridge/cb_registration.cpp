#include "cb_registration.hpp"
#include "utils.hpp"
#include "pybridge.h"

namespace py = pybind11;

namespace pybridge {

Callback* g_callbacks[static_cast<int>(CallbackEnum::Count)] = {nullptr};

static int set_cb_ptr(CallbackEnum which, bool unset)
{
    switch (which) {
        case CallbackEnum::GetDeltaQOffset:
            get_deltaq_offset_cb = unset ? nullptr : get_deltaq_offset_trampoline;
            return 0;
        case CallbackEnum::RecvPictureFeedback:
            recv_picture_feedback_cb = unset ? nullptr : recv_picture_feedback_trampoline;
            return 0;
        default:
            return -1;
    }
}

int pybridge_set_cb(CallbackEnum which, py::object callable)
{
    Callback &slot = *g_callbacks[static_cast<int>(which)];

    if (callable.is_none()) {
        // Unset function
        slot.py_func = std::move(py::none());
    }
    else {
        // Validate and store new function
        py::function function =  pyutils::validate_callable(callable, slot.n_args);
        slot.py_func = std::move(function);
    }

    // Enable the C trampoline for the encoder
    return set_cb_ptr(which, callable.is_none());
}

} // namespace bridge
