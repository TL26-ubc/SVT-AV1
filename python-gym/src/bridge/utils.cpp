#include "utils.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <Python.h>

#include <cassert>

namespace pyutils {

py::function validate_callable(const py::object &callable, int expected_n_args)
{
    assert(PyGILState_Check());

    if (!PyCallable_Check(callable.ptr())) {
        throw py::type_error("object must be callable");
    }

    // verify __code__ object
    py::object code_obj = callable.attr("__code__");
    if (!PyCode_Check(code_obj.ptr())) {
        throw py::type_error("callable has no valid __code__");
    }

    // Grab arg numbers
    PyCodeObject *co = reinterpret_cast<PyCodeObject *>(code_obj.ptr());
    const int nargs       = co->co_argcount + co->co_posonlyargcount;
    const bool has_vararg = (co->co_flags & CO_VARARGS)     != 0;
    const bool has_varkw  = (co->co_flags & CO_VARKEYWORDS) != 0;

    if (has_vararg || has_varkw) {
        throw py::type_error("callback may not use *args or **kwargs");
    }

    if (nargs != expected_n_args) {
        throw py::type_error(
            "callback must take exactly " + std::to_string(expected_n_args) + " positional arguments (got " + std::to_string(nargs) + ")"
        );
    }

    // Return as a function
    return py::reinterpret_borrow<py::function>(callable);
}

}