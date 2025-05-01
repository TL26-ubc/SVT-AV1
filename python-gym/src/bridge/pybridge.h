#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#include <Python.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Global function pointer the encoder’s hot code will call.          */
/*  Each SVT worker thread may invoke this, so the target MUST         */
/*  acquire the GIL in its trampoline (implemented in pybridge.c).     */
extern int (*g_py_sb_qp_cb)(int row, int col,
                            uint64_t stat1, uint64_t stat2);

void pybridge_set_cb(PyObject *callable);
void pybridge_clear(void);

#endif /* PYBRIDGE_H */
