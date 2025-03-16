#ifndef TL26_FLAGS_H
#define TL26_FLAGS_H

// #undef TL26_RL
#define TL26_RL

#ifdef TL26_RL
#include <Python.h>

extern PyObject *f_frame_report_feedback, *f_sb_send_offset_request;

void initialize_python();
void finalize_python();

#endif

#endif // TL26_FLAGS_H