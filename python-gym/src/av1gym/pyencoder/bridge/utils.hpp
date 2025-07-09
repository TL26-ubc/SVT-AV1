#pragma once
#include <pybind11/pybind11.h>

namespace py  = pybind11;

namespace pyutils {

py::function validate_callable(const py::object &callable, int expected_n_args);

template <typename T, typename Transform>
py::list to_pylist(const T *data, std::size_t n, Transform &&fn)
{
    if (!data)
        throw py::value_error("data pointer is null");

    py::list out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = fn(data[i]);
    return out;
}

} // namespace pybridge
