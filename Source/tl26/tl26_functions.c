#include "tl26_flags.h"
#include "tl26_python_thread.h"

#ifdef TL26_RL
static PyObject *f_frame_report_feedback, *f_sb_send_offset_request;
static PyThreadState *main_thread_state = NULL;

void initialize_python() {
    Py_Initialize();
    main_thread_state = PyThreadState_Get();
    PyEval_ReleaseThread(main_thread_state);
    init_python_thread();
}

void finalize_python() {
    set_python_thread_running(false);
    signal_python_thread_termination();
    shutdown_python_thread();
    
    usleep(300000);
    
    PyEval_AcquireThread(main_thread_state);
    
    Py_Finalize();
}

PyObject *get_f_frame_report_feedback() {
    return f_frame_report_feedback;
}

PyObject *get_f_sb_send_offset_request() {
    return f_sb_send_offset_request;
}
#endif