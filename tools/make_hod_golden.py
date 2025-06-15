#!/usr/bin/env python3
"""Regenerate (or check) hod_golden.dat by an independent Taylor jet.

The golden file records T^{(N,alpha)}_1 for every carried multi-index at
one input point, and the trainer's [REF] line compares its own tables
against it.  That comparison is only worth something if the two sides
are independent, so this script does not use the library: it propagates
truncated multivariate Taylor series through the same network by hand,
in Python, with the coefficients read from the network's weight file and
the multi-index order read from hod_alpha_order.dat.

    python3 tools/make_hod_golden.py CASE_DIR            # check
    python3 tools/make_hod_golden.py CASE_DIR --write    # regenerate

Run it from the repository root, with CASE_DIR a directory that already
holds nn_weight.dat, hod_alpha_order.dat and hod_golden.dat (any case
run once with Hod_check 1 and Activation TANH).  Only the SHAPE is taken
from the weight file: the check overwrites the weights with a
deterministic formula before evaluating, so the golden file is a
property of the architecture and not of any run, and this script
reproduces that formula.  Needs only the standard library.

Measured against the shipped files: 2.2e-19 over the 330 slots of
zwork/hod_4d_k7 and 2.8e-17 over the 35 of zwork/hod_4d_k3.
"""
import itertools
import math
import os
import sys


def read_weights(path):
    """Layer dimensions from a DNNF90 weight file, and the CHECK weights.

    The self-check does not evaluate the trained weights: it overwrites
    them with a deterministic formula first (hod_check_module.f90), so
    that the golden file is a property of the architecture alone and not
    of any particular run.  This reproduces that formula, and takes only
    the shape from the weight file.
    """
    tok = open(path).read().split()
    p = 0
    p += 1                                   # epoch
    assert tok[p] == "func"; func = int(tok[p + 1]); p += 2
    assert tok[p] == "Activation_out"; p += 2
    nlayer = int(tok[p]); p += 1
    dims = [int(tok[p + i]) for i in range(nlayer)]; p += nlayer
    w = [None] * nlayer
    for l in range(1, nlayer):
        # Fortran indices: l runs 2..Nlayer, j runs 1..ndim(l),
        # i runs 0..ndim(l-1).  Python's l is one less than Fortran's.
        rows = []
        for jn in range(1, dims[l] + 1):
            rows.append([0.1 * math.sin(1.7 * (l + 1) + 0.9 * jn + 0.3 * i) + 0.05
                         for i in range(dims[l - 1] + 1)])
        w[l] = rows
    return dims, w, func


def read_alpha(path):
    """The carried multi-indices, in the order the tables use."""
    # columns: ia, |alpha|, then alpha(1:D0) -- drop the first two
    out = []
    for line in open(path):
        f = line.split()
        if not f or not f[0].lstrip("-").isdigit():
            continue
        out.append(tuple(int(v) for v in f[2:]))
    return out


class Jet:
    """A truncated multivariate Taylor polynomial, coefficients by index."""

    def __init__(self, d0, k, c=None):
        self.d0, self.k = d0, k
        self.c = dict(c) if c else {}

    def _keys(self):
        for deg in range(self.k + 1):
            for comb in itertools.combinations_with_replacement(range(self.d0), deg):
                a = [0] * self.d0
                for t in comb:
                    a[t] += 1
                yield tuple(a)

    def __add__(self, o):
        r = Jet(self.d0, self.k, self.c)
        for a, v in o.c.items():
            r.c[a] = r.c.get(a, 0.0) + v
        return r

    def scale(self, s):
        return Jet(self.d0, self.k, {a: v * s for a, v in self.c.items()})

    def shift(self, s):
        r = Jet(self.d0, self.k, self.c)
        z = (0,) * self.d0
        r.c[z] = r.c.get(z, 0.0) + s
        return r

    def __mul__(self, o):
        r = Jet(self.d0, self.k)
        for a, va in self.c.items():
            if va == 0.0:
                continue
            for b, vb in o.c.items():
                s = tuple(x + y for x, y in zip(a, b))
                if sum(s) > self.k:
                    continue
                r.c[s] = r.c.get(s, 0.0) + va * vb
        return r

    def compose_tanh(self):
        """tanh of a jet, by the ODE y' = 1 - y^2 on the series."""
        z = (0,) * self.d0
        a0 = self.c.get(z, 0.0)
        # Taylor coefficients of tanh at a0, to order k
        t = [math.tanh(a0)]
        # d/da tanh = 1 - t^2 ; build derivatives symbolically in t
        poly = [[0.0, 0.0, -1.0]]           # P_1(t) = 1 - t^2 stored as coeffs
        poly[0] = [1.0, 0.0, -1.0]
        for q in range(1, self.k + 1):
            pv = sum(poly[q - 1][i] * t[0] ** i for i in range(len(poly[q - 1])))
            t.append(pv)
            # P_{q+1} = (1 - t^2) * dP_q/dt
            dp = [i * poly[q - 1][i] for i in range(1, len(poly[q - 1]))]
            nxt = [0.0] * (len(dp) + 2)
            for i, v in enumerate(dp):
                nxt[i] += v
                nxt[i + 2] -= v
            poly.append(nxt)
        # substitute the deviation series into the univariate expansion
        dev = Jet(self.d0, self.k, {a: v for a, v in self.c.items() if a != z})
        out = Jet(self.d0, self.k, {z: t[0]})
        powr = Jet(self.d0, self.k, {z: 1.0})
        for q in range(1, self.k + 1):
            powr = powr * dev
            out = out + powr.scale(t[q] / math.factorial(q))
        return out


def net_jet(dims, w, x, k):
    d0 = dims[0]
    cur = []
    for i in range(d0):
        j = Jet(d0, k, {(0,) * d0: x[i]})
        e = [0] * d0
        e[i] = 1
        j.c[tuple(e)] = 1.0
        cur.append(j)
    for l in range(1, len(dims)):
        nxt = []
        for jn in range(dims[l]):
            # the weight file stores w(l,j,0:ndim(l-1)): index 0 is the
            # bias, 1..ndim(l-1) are the incoming weights
            s = Jet(d0, k, {(0,) * d0: w[l][jn][0]})
            for i in range(dims[l - 1]):
                s = s + cur[i].scale(w[l][jn][i + 1])
            nxt.append(s if l == len(dims) - 1 else s.compose_tanh())
        cur = nxt
    return cur[0]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    d = sys.argv[1]
    write = "--write" in sys.argv
    dims, w, func = read_weights(os.path.join(d, "nn_weight.dat"))
    if func != 0:
        print("this generator implements tanh only (func=%d in the file)" % func)
        sys.exit(2)
    alphas = read_alpha(os.path.join(d, "hod_alpha_order.dat"))
    gold = os.path.join(d, "hod_golden.dat")
    x = [float(v) for v in open(gold).readline().split()]
    k = max(sum(a) for a in alphas)
    jet = net_jet(dims, w, x, k)
    # T^alpha = alpha! * (Taylor coefficient)
    vals = []
    for a in alphas:
        fac = 1.0
        for v in a:
            fac *= math.factorial(v)
        vals.append(jet.c.get(a, 0.0) * fac)
    if write:
        with open(gold, "w") as fh:
            fh.write(" ".join("%.17e" % v for v in x) + "\n")
            for i, v in enumerate(vals, 1):
                fh.write("%d %.17e\n" % (i, v))
        print("wrote %s (%d slots)" % (gold, len(vals)))
        return
    ref = {}
    for line in open(gold).readlines()[1:]:
        f = line.split()
        if len(f) == 2:
            ref[int(f[0])] = float(f[1])
    emax = 0.0
    for i, v in enumerate(vals, 1):
        if i in ref:
            emax = max(emax, abs(v - ref[i]))
    print("%d slots, max |jet - golden| = %.3e" % (len(vals), emax))
    print("passed" if emax < 1e-10 else "FAILED")
    sys.exit(0 if emax < 1e-10 else 1)


main()
