#!/usr/bin/env python3
# tools/example_py.py - DNNF90 called from Python via ctypes, the way an
# ASE calculator or an i-PI driver would.  (MIT License.)
#
# Build the shared library first, then run in a trained benchmark dir:
#     make shared
#     cd bench/kdv && python3 ../../tools/example_py.py
import ctypes, os, sys

here = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(os.path.join(here, "..", "libdnnf90.so"))

lib.dnnf90_tables_init_closure.argtypes = [ctypes.c_int]*3 + [ctypes.POINTER(ctypes.c_int)]
lib.dnnf90_net_load.argtypes = [ctypes.c_char_p]
lib.dnnf90_eval.argtypes = [ctypes.c_int, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double)]

seeds = (ctypes.c_int * 6)(0, 1, 1, 0, 3, 0)      # u_t, u_x, u_xxx
assert lib.dnnf90_tables_init_closure(2, 3, 3, seeds) == 0
n = lib.dnnf90_nderiv()
nid = lib.dnnf90_net_load(b"nn_weight.dat")
assert nid > 0, "run in a trained benchmark directory"
wid = lib.dnnf90_work_create(nid)

x = (ctypes.c_double * 2)(0.0, 0.5)
t = (ctypes.c_double * 64)()
lib.dnnf90_eval(nid, wid, x, t)
print(f"slots={n}  u(0,0.5)={t[0]:.6f}  (expected about 0.9427)")
print("python -> C ABI -> Fortran: OK")
