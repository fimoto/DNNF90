#!/usr/bin/env python3
"""How far are the network's seventh derivatives from the exact soliton?

Section 6 of the manuscript states that the seventh-order ZK run reaches
0.06% solution error while its seventh derivatives exceed the exact
values manyfold.  A collocation run writes no high-order derivatives, so
this script reads them from tools/hod_dump.out and compares slot by slot
against the exact line soliton u = A sech^{2m}(k(x + l y + n z - c t)).

    make hod_dump.out
    cd bench/zk7
    ../../build/hod_dump.out nn_weight.dat colloc.dat 4 7 5 > hod_dump.dat
    python3 ../post/zk7_seventh.py hod_dump.dat

Options (defaults are the shipped bench/zk7 case):
    --k 0.15 --m 0.5 --n 0.3333333333333333 --c 2.6244 --power 6

It prints three ratios, because "exceeds the exact value" depends on
which slots are counted: over the ten seventh-order slots the residual
actually uses, over all 120 seventh-order slots, and the worst single
slot.  Quote whichever the text means.
"""
import argparse
import sys

import numpy as np


def read_dump(path):
    alphas, rows = [], []
    for line in open(path):
        if line.startswith("# alpha"):
            alphas.append(tuple(int(v) for v in line.split(":")[1].split()))
        elif line.startswith("#") or line.startswith("###"):
            continue
        elif line.strip():
            rows.append([float(v) for v in line.split()])
    return alphas, np.array(rows)


def sech_derivs(power, order):
    """symbolic d^q/dxi^q of sech^power, as callables"""
    import sympy as sp
    xi = sp.symbols("xi")
    u = sp.sech(xi) ** power
    return [sp.lambdify(xi, sp.diff(u, xi, q).rewrite(sp.exp), "numpy")
            for q in range(order + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--k", type=float, default=0.15)
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--n", type=float, default=1.0 / 3.0)
    ap.add_argument("--c", type=float, default=2.6244)
    ap.add_argument("--power", type=int, default=6)
    ap.add_argument("--order", type=int, default=7)
    a = ap.parse_args()

    alphas, D = read_dump(a.dump)
    d0 = len(alphas[0])
    X, T = D[:, :d0], D[:, d0:]
    if T.shape[1] != len(alphas):
        sys.exit("column count does not match the alpha header")

    Ud = sech_derivs(a.power, a.order)
    arg = a.k * (X[:, 0] + a.m * X[:, 1] + a.n * X[:, 2] - a.c * X[:, 3])

    def exact(al):
        p = sum(al)
        return (a.k ** p * a.m ** al[1] * a.n ** al[2] * (-a.c) ** al[3]
                * Ud[p](arg))

    print("order   slots   ||net||/||exact||   ||net-exact||/||exact||")
    for p in range(a.order + 1):
        idx = [i for i, al in enumerate(alphas) if sum(al) == p]
        E = np.array([exact(alphas[i]) for i in idx]).T
        N = T[:, idx]
        print("%5d %7d %19.3f %24.3f"
              % (p, len(idx), np.linalg.norm(N) / np.linalg.norm(E),
                 np.linalg.norm(N - E) / np.linalg.norm(E)))

    top = [i for i, al in enumerate(alphas) if sum(al) == a.order]
    worst = max(top, key=lambda i: (np.abs(T[:, i]).max()
                                    / max(np.abs(exact(alphas[i])).max(), 1e-300)))
    r = (np.abs(T[:, worst]).max()
         / np.abs(exact(alphas[worst])).max())
    print("\nworst order-%d slot %s: max|net| / max|exact| = %.1f"
          % (a.order, alphas[worst], r))


if __name__ == "__main__":
    main()
