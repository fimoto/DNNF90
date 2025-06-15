#!/usr/bin/env python3
"""Does the System block of a Lax case reproduce the equation?

    python3 bench/post/check_lax_input.py zwork/pinn_lax
    python3 bench/post/check_lax_input.py zwork/pinn_lax7 --order 7

Reads input_nn.dat independently of the Fortran parser, evaluates the
declared residual at the exact soliton u = 2k^2 sech^2(k(x-16k^4 t))
with symbolically exact derivatives, and reports the largest residual
over the collocation points. A correct table leaves rounding.
"""
import os, sys
import numpy as np
import sympy as sp

def parse_system(path, d0):
    terms = []
    lines = open(path).read().split("\n")
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if w and w[0].lower() == "system":
            nres, nterm = int(w[1]), int(w[2])
            got = 0
            while got < nterm:
                i += 1
                t = lines[i].split()
                if not t or t[0].startswith("#"):
                    continue
                kind = t[0].upper()
                if kind == "TRM":
                    ir, jc = int(t[1]), int(t[2])
                    al = tuple(int(v) for v in t[3:3+d0])
                    c = float(t[3+d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al]))
                elif kind == "XUX":
                    ir, ic, jc = int(t[1]), int(t[2]), int(t[3])
                    al = tuple(int(v) for v in t[4:4+d0])
                    c = float(t[4+d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al, tuple([0]*d0)]))
                elif kind == "DXD":
                    ir, ic = int(t[1]), int(t[2])
                    be = tuple(int(v) for v in t[3:3+d0])
                    jc = int(t[3+d0])
                    al = tuple(int(v) for v in t[4+d0:4+2*d0])
                    c = float(t[4+2*d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al, be]))
                elif kind == "QAD":
                    ir, lc = int(t[1]), int(t[2])
                    de = tuple(int(v) for v in t[3:3+d0])
                    kc = int(t[3+d0])
                    ga = tuple(int(v) for v in t[4+d0:4+2*d0])
                    ic = int(t[4+2*d0])
                    be = tuple(int(v) for v in t[5+2*d0:5+3*d0])
                    jc = int(t[5+3*d0])
                    al = tuple(int(v) for v in t[6+3*d0:6+4*d0])
                    c = float(t[6+4*d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al, be, ga, de]))
                elif kind == "QAD":
                    ir, lc = int(t[1]), int(t[2])
                    de = tuple(int(v) for v in t[3:3+d0])
                    kc = int(t[3+d0])
                    ga = tuple(int(v) for v in t[4+d0:4+2*d0])
                    ic = int(t[4+2*d0])
                    be = tuple(int(v) for v in t[5+2*d0:5+3*d0])
                    jc = int(t[5+3*d0])
                    al = tuple(int(v) for v in t[6+3*d0:6+4*d0])
                    c = float(t[6+4*d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al, be, ga, de]))
                elif kind == "TRP":
                    ir, kc = int(t[1]), int(t[2])
                    ga = tuple(int(v) for v in t[3:3+d0])
                    ic = int(t[3+d0])
                    be = tuple(int(v) for v in t[4+d0:4+2*d0])
                    jc = int(t[4+2*d0])
                    al = tuple(int(v) for v in t[5+2*d0:5+3*d0])
                    c = float(t[5+3*d0].replace("d","e").replace("D","e"))
                    terms.append((ir, c, [al, be, ga]))
                else:
                    sys.exit("unhandled term type " + kind)
                got += 1
            return nres, terms
        i += 1
    sys.exit("no System block found")

def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    order = 7 if "--order" in " ".join(sys.argv) and "7" in sys.argv else 5
    d = argv[0] if argv else "."
    nres, terms = parse_system(os.path.join(d, "input_nn.dat"), 2)
    print("  %d residual(s), %d terms" % (nres, len(terms)))

    x, t = sp.symbols("x t")
    k = sp.Rational(1, 2)
    if "lax7" in os.path.abspath(d):
        # Chen et al., Sci. Rep. 14 (2024) 23874, Eq. (39)
        u = sp.Rational(1, 2) / sp.cosh((x - t)/2)**2
    else:
        u = 2*k**2 / sp.cosh(k*(x - 16*k**4*t))**2
    cache = {}
    def dfun(a):
        if a not in cache:
            e = sp.diff(u, x, a[0], t, a[1])
            cache[a] = sp.lambdify((x, t), e, "numpy")
        return cache[a]

    rows = np.loadtxt(os.path.join(d, "colloc.dat"))
    X, T = rows[:, 0], rows[:, 1]
    R = np.zeros((nres, len(X)))
    for ir, c, alphas in terms:
        p = c * np.ones_like(X)
        for a in alphas:
            p = p * dfun(a)(X, T)
        R[ir-1] += p
    worst = np.abs(R).max(axis=1)
    bad = False
    for kk in range(nres):
        flag = "" if worst[kk] < 1e-10 else "   <-- WRONG"
        bad = bad or worst[kk] >= 1e-10
        print("  residual %d: max |R(exact)| = %.3e%s" % (kk+1, worst[kk], flag))
    print("  " + ("FAILED" if bad else "the table reproduces the equation"))
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
