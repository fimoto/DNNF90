#!/usr/bin/env python3
"""Does the Kovasznay term table reproduce the Navier-Stokes residual?

    python3 bench/post/check_kovasznay.py <case dir>

Reads the System block as the input file declares it, evaluates every
term at the exact solution, and checks that the residuals vanish.  The
Kovasznay flow is an exact solution, so they must, and a sign or a
component index out of place would otherwise show up only as a fit that
does not converge.
"""
import math
import os
import sys

NU = 0.025
LAM = 1/(2*NU) - math.sqrt(1/(4*NU*NU) + 4*math.pi**2)


def exact(x, y, comp, a):
    """derivative a = (ax, ay) of component comp at (x, y)"""
    e = math.exp(LAM*x)
    k = 2*math.pi
    ax, ay = a
    if comp == 1:                       # u = 1 - e^(lam x) cos(k y)
        if (ax, ay) == (0, 0):
            return 1 - e*math.cos(k*y)
        c = -(LAM**ax)*e
        if ay % 2 == 0:
            return c*((-1)**(ay//2))*(k**ay)*math.cos(k*y)
        return c*((-1)**((ay+1)//2))*(k**ay)*math.sin(k*y)
    if comp == 2:                       # v = lam/k e^(lam x) sin(k y)
        c = (LAM/k)*(LAM**ax)*e
        if ay % 2 == 0:
            return c*((-1)**(ay//2))*(k**ay)*math.sin(k*y)
        return c*((-1)**(ay//2))*(k**ay)*math.cos(k*y)
    if comp == 3:                       # p = (1 - e^(2 lam x))/2
        if ay > 0:
            return 0.0
        if ax == 0:
            return (1 - math.exp(2*LAM*x))/2
        return -0.5*(2*LAM)**ax*math.exp(2*LAM*x)
    raise ValueError(comp)


def parse(path, d0):
    terms = []
    lines = open(path).read().split("\n")
    i = 0
    nres = 0
    while i < len(lines):
        w = lines[i].split()
        if w and w[0].lower() == "system":
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
                    al = tuple(int(q) for q in t[3:3+d0])
                    c = float(t[3+d0].replace("d", "e"))
                    terms.append((ir, c, [(jc, al)]))
                elif kind == "XUX":
                    ir, ic, jc = int(t[1]), int(t[2]), int(t[3])
                    al = tuple(int(q) for q in t[4:4+d0])
                    c = float(t[4+d0].replace("d", "e"))
                    terms.append((ir, c, [(jc, al), (ic, (0,)*d0)]))
                else:
                    sys.exit("unhandled term " + kind)
                got += 1
        i += 1
    return nres, terms


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    nres, terms = parse(os.path.join(d, "input_nn.dat"), 2)
    print("  lambda = %.8f, %d residuals, %d terms" % (LAM, nres, len(terms)))
    worst = [0.0]*nres
    for x, y in ((0.3, 0.7), (-0.2, 0.4), (0.8, -0.1), (0.55, 1.2)):
        R = [0.0]*nres
        for ir, c, facs in terms:
            q = c
            for comp, a in facs:
                q *= exact(x, y, comp, a)
            R[ir-1] += q
        for k in range(nres):
            worst[k] = max(worst[k], abs(R[k]))
    names = ["continuity", "x-momentum", "y-momentum"]
    bad = False
    for k in range(nres):
        flag = "" if worst[k] < 1e-10 else "   <-- WRONG"
        if worst[k] >= 1e-10:
            bad = True
        print("    R%d %-12s %.4e%s" % (k+1, names[k], worst[k], flag))
    if bad:
        sys.exit("the term table does not reproduce the equations")
    print("  the term table reproduces the equations")


if __name__ == "__main__":
    main()
