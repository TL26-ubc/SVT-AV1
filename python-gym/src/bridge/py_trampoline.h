#ifndef PY_TRAMPOLINE_H
#define PY_TRAMPOLINE_H

#include <Python.h>

/*  py_trampoline
 *  -------------
 *  Call a Python callable from C with typed arguments.
 *
 *  Usage
 *  -----
 *      rc = py_trampoline(cb, "(fmt)r", &ret, …args…);
 *
 *      • “(fmt)”  – one or more **argument codes**
 *      • “r”      – **one return‑code**
 *      • &ret     – pointer to C storage for the result
 *      • …args…   – C values that match the codes in order
 *
 *  Argument codes
 *  --------------
 *      Integers (uppercase unsigned)
 *          c/C 8‑bit   
 *          h/H 16‑bit   
 *          i/u 32‑bit   
 *          q/Q 64‑bit
 *      Other scalar 
 *          d double    
 *          b bool/int   
 *          s const char*   
 *          O PyObject*
 *      Pointer+fn   
 *          T void*, PyObject *(*transform)(void*)
 *
 *      Containers   
 *          Lx  ist (buffer, len [ ,transform ])
 *          Mx matrix (buffer, w ,h [ ,transform ])
 *          *x is any element code above (including T).
 *
 *  Return codes
 *  ------------
 *      i 32‑bit int   
 *      u 32‑bit unsigned   
 *      d double   
 *      b bool
 *      O PyObject*             
 *      v void / ignored
 *      L list of 32-bit int
 *
 *  Result
 *  ------
 *      Returns 0 on success, ‑1 on error (Python exception is set).
 */
int py_trampoline(PyObject *cb, const char *fmt, void* ret, ...);

#endif /* PY_TRAMPOLINE_H */