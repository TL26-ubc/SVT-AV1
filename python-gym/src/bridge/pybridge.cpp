#include "cb_registration.hpp"
#include "utils.hpp"
#include "pybridge.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybridge;

get_deltaq_offset_cb_t     get_deltaq_offset_cb     = nullptr;
recv_picture_feedback_cb_t recv_picture_feedback_cb = nullptr;

extern "C" void get_deltaq_offset_trampoline(int *offset_array, uint32_t sb_count, int32_t picture_number) {
    Callback &cb = *g_callbacks[static_cast<int>(CallbackEnum::GetDeltaQOffset)];
    if (cb.py_func.is_none())
        return;

    py::gil_scoped_acquire acquire;

    py::function fcn = pyutils::validate_callable(cb.py_func, cb.n_args);

    py::object ret = fcn.operator()(picture_number);

    if (!py::isinstance<py::list>(ret)) {
        throw py::type_error("Expected return value of type list however was " +
                             py::cast<std::string>(ret.get_type().attr("__name__")));
    }

    py::list qp_map = py::cast<py::list>(ret);
    if (qp_map.size() != sb_count) {
        throw py::value_error("Expected return value of type list with size " + std::to_string(sb_count) +
                              " however was of size " + std::to_string(qp_map.size()));
    }

    for (uint32_t i = 0; i < sb_count; ++i) {
        py::object item = qp_map[i];

        if (!py::isinstance<py::int_>(item)) {
            throw py::type_error("qp_map[" + std::to_string(i) + "] is not an int, got " +
                                 py::cast<std::string>(item.get_type().attr("__name__")));
        }

        offset_array[i] = item.cast<int>();
    }
}

extern "C" void recv_picture_feedback_trampoline(uint8_t *bitstream, uint32_t bitstream_size, uint32_t picture_number) {
    Callback &cb = *g_callbacks[static_cast<int>(CallbackEnum::RecvPictureFeedback)];
    if (cb.py_func.is_none())
        return;

    py::gil_scoped_acquire acquire;

    py::function fcn = pyutils::validate_callable(cb.py_func, cb.n_args);

    py::bytes py_bitstream = py::bytes(reinterpret_cast<char *>(bitstream), bitstream_size);

    fcn.operator()(py_bitstream, bitstream_size, picture_number);
}
