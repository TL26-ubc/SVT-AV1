#ifndef TL26_FLAGS_H
#define TL26_FLAGS_H

// #undef TL26_RL
#define TL26_RL

#ifdef TL26_RL
#include <Python.h>

// PyObject *f_frame_report_feedback, *f_sb_send_offset_request;

void initialize_python();
void finalize_python();

PyObject *get_f_frame_report_feedback();
PyObject *get_f_sb_send_offset_request();
#endif

#endif // TL26_FLAGS_H