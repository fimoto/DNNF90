#!/usr/bin/env python3
"""Does the System block as parsed reproduce the equations?

    python3 bench/post/check_system_input.py <case dir>

The term table of a case is written by hand, and a sign or a component
index out of place makes the residual wrong in a way that shows up only
as a fit that will not converge.  This reads the collocation file, which
carries the exact solution the case was manufactured from, and evaluates
the residual of that solution using the term table as the input file
declares it.  Every residual should come out at the source value, so the
difference should be zero to rounding.

It duplicates the Fortran evaluation deliberately: agreeing with an
independent reading of the same input file is the point.
"""
import math
import os
import re
import sys


def parse_input(path):
    """the System block, the network shape, and the loss terms"""
    lines = open(path).read().split("\n")
    d0 = nout = None
    terms = []
    nres = 0
    i = 0
    while i < len(lines):
        w = lines[i].split()
        if w and w[0].lower() == "nlayer":
            nl = int(w[1])
            dims = []
            for k in range(nl):
                i += 1
                while not lines[i].split() or lines[i].lstrip().startswith("#"):
                    i += 1
                dims.append(int(lines[i].split()[0]))
            d0, nout = dims[0], dims[-1]
        elif w and w[0].lower() == "system":
            nres, nterm = int(w[1]), int(w[2])
            got = 0
            while got < nterm:
                i += 1
                t = lines[i].split("/")[0].split()
                if not t or lines[i].lstrip().startswith("#"):
                    continue
                kind = t[0].upper()
                if kind == "TRM":
                    ir, jc = int(t[1]), int(t[2])
                    al = [int(v) for v in t[3:3+d0]]
                    c = float(t[3+d0].replace("d", "e"))
                    terms.append((ir, c, [(jc, tuple(al))]))
                elif kind == "XUX":
                    ir, ic, jc = int(t[1]), int(t[2]), int(t[3])
                    al = [int(v) for v in t[4:4+d0]]
                    c = float(t[4+d0].replace("d", "e"))
                    terms.append((ir, c, [(jc, tuple(al)),
                                          (ic, tuple([0]*d0))]))
                elif kind == "DXD":
                    ir, ic = int(t[1]), int(t[2])
                    be = [int(v) for v in t[3:3+d0]]
                    jc = int(t[3+d0])
                    al = [int(v) for v in t[4+d0:4+2*d0]]
                    c = float(t[4+2*d0].replace("d", "e"))
                    terms.append((ir, c, [(jc, tuple(al)), (ic, tuple(be))]))
                else:
                    sys.exit("unhandled term type " + kind)
                got += 1
        i += 1
    return d0, nout, nres, terms


# ---- the exact solution of the electrohydrodynamic case ---------------
def exact(x, y, comp, a):
    """derivative a of component comp, by hand"""
    s, c = math.sin(x), math.cos(x)
    S, C = math.sin(y), math.cos(y)

    def phi(ax, ay):
        t = [[s*S, s*C, -s*S], [c*S, c*C, -c*S], [-s*S, -s*C, s*S]]
        return t[ax][ay]

    if comp == 1:
        return phi(a[0], a[1])
    if comp == 2:                       # rho = cos x cos y + 1/2
        t = [[c*C, -c*S, -c*C], [-s*C, s*S, s*C], [-c*C, c*S, c*C]]
        r = t[a[0]][a[1]]
        if a == (0, 0):
            r += 0.5
        return r
    if comp == 3:                       # u = -cos x sin y
        t = [[-c*S, -c*C, c*S], [s*S, s*C, -s*S], [c*S, c*C, -c*S]]
        return t[a[0]][a[1]]
    if comp == 4:                       # v = sin x cos y
        t = [[s*C, -s*S, -s*C], [c*C, -c*S, -c*C], [-s*C, s*S, s*C]]
        return t[a[0]][a[1]]
    if comp == 5:                       # p = -(cos2x + cos2y)/4
        if a == (0, 0):
            return -(math.cos(2*x) + math.cos(2*y))/4.0
        if a == (1, 0):
            return math.sin(2*x)/2.0
        if a == (0, 1):
            return math.sin(2*y)/2.0
        if a == (2, 0):
            return math.cos(2*x)
        if a == (0, 2):
            return math.cos(2*y)
        return 0.0
    raise ValueError(comp)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    d0, nout, nres, terms = parse_input(os.path.join(d, "input_nn.dat"))
    print("  %d inputs, %d components, %d residuals, %d terms"
          % (d0, nout, nres, len(terms)))

    rows = [l.split() for l in open(os.path.join(d, "colloc.dat"))
            if l.strip() and not l.lstrip().startswith("#")]
    worst = [0.0]*nres
    for r in rows[:200]:
        x, y = float(r[0]), float(r[1])
        src = [float(v) for v in r[d0:d0+nres]]
        R = [0.0]*nres
        for ir, c, facs in terms:
            p = c
            for comp, a in facs:
                p *= exact(x, y, comp, a)
            R[ir-1] += p
        for k in range(nres):
            worst[k] = max(worst[k], abs(R[k] - src[k]))

    print("  residual of the exact solution minus its source, per residual:")
    bad = False
    for k in range(nres):
        flag = "" if worst[k] < 1e-10 else "   <-- WRONG"
        if worst[k] >= 1e-10:
            bad = True
        print("    R%d  %.4e%s" % (k+1, worst[k], flag))
    if bad:
        sys.exit("the term table does not reproduce the equations")
    print("  the term table reproduces the equations")


if __name__ == "__main__":
    main()
