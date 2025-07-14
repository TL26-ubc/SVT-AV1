extern "C" {
    #define main app_main
    #include "../Source/App/app_main.c"
    #undef main
}

#ifndef SVT_ENABLE_USER_CALLBACKS
#define SVT_ENABLE_USER_CALLBACKS 1
#endif

#include <pybind11/pybind11.h>

#include "../bridge/utils.hpp"
#include "../bridge/cb_registration.hpp"
#include "../bridge/pybridge.h"

#include "../Source/Lib/Globals/enc_callbacks.h"
#include "../Source/API/EbSvtAv1Enc.h"

#include <vector>
#include <string>

namespace py = pybind11;
using namespace pybridge;

int init_callbacks();
void deinit_callbacks();

// run(argv: List[str]) -> None
static py::object run(py::list py_argv)
{
    const int argc = static_cast<int>(py_argv.size());

    // Keep string storage alive for the whole call.
    std::vector<std::string> storage;
    storage.reserve(argc);

    std::vector<char *> argv;
    argv.reserve(argc + 1);

    // Parse args to argv and argc list
    for (const py::handle &item : py_argv) {
        storage.emplace_back(py::cast<std::string>(item));
        argv.push_back(const_cast<char *>(storage.back().c_str()));
    }
    argv.push_back(nullptr);

    int rc = 0;
    {
        // Release the GIL while the encoder CLI runs.
        py::gil_scoped_release release;
        rc = app_main(argc, argv.data());
        py::gil_scoped_acquire acquire;
        deinit_callbacks();
    }

    if (rc != 0) {
        throw std::runtime_error(
            "SvtAv1EncApp returned non‑zero exit code " + std::to_string(rc));
    }

    return py::none();
}

// register_callbacks(get_deltaq_offset=None, picture_feedback=None, postencode_feedback=None) -> None
static py::object register_callbacks(py::object py_get_deltaq_offset = py::none(),
                                     py::object py_picture_feedback = py::none(),
                                     py::object py_post_encode_feedback = py::none())
{
    init_callbacks();

    // Store the user callables and hook up the C trampolines
    pybridge_set_cb(CallbackEnum::GetDeltaQOffset, py_get_deltaq_offset);
    pybridge_set_cb(CallbackEnum::RecvPictureFeedback, py_picture_feedback);
    pybridge_set_cb(CallbackEnum::RecvPostEncodeStats, py_post_encode_feedback);

    // Tell SVT‑AV1 about the trampolines
    static PluginCallbacks cbs;
    cbs.user_get_deltaq_offset = get_deltaq_offset_cb;
    cbs.user_picture_feedback = recv_picture_feedback_cb;
    cbs.user_postencode_feedback = recv_postencode_feedback_cb;

    if (svt_av1_enc_set_callbacks(&cbs) != EB_ErrorNone) {
        throw std::runtime_error("failed to set callbacks");
    }

    return py::none();
}

PYBIND11_MODULE(_av1_wrapper, m)
{
    m.doc() = "In‑process bindings for the SVT‑AV1 encoder CLI";

    m.def("run", &run,
          "Run the SVT‑AV1 encoder CLI in‑process.");

    m.def("register_callbacks", &register_callbacks,
          py::arg("get_deltaq_offset") = py::none(),
          py::arg("picture_feedback")  = py::none(),
          py::arg("postencode_feedback")  = py::none(),
          "Attach callbacks to the SVT‑AV1 encoder.");
}

int init_callbacks()
{
    for (int i = 0; i < static_cast<int>(CallbackEnum::Count); ++i) {
        g_callbacks[i] = new Callback{py::none(), nullptr, 0};
    }
    g_callbacks[0]->n_args = 6; // GetDeltaQOffset
    g_callbacks[1]->n_args = 3; // RecvPictureFeedback
    g_callbacks[2]->n_args = 2; // RecvPostEncodeStats
    return 0;
}

void deinit_callbacks()
{
    for (int i = 0; i < static_cast<int>(CallbackEnum::Count); ++i) {
        delete g_callbacks[i];
        g_callbacks[i] = nullptr;
    }
}