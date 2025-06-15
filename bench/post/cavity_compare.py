#!/usr/bin/env python3
"""Compare a lid-driven cavity PINN with the Ghia reference profiles.

    python3 bench/post/cavity_compare.py <run dir> [out.pdf]

The centreline velocities of the lid-driven cavity at Re = 100 are the
standard check for a flow solver, and Ghia, Ghia and Shin (1982) are the
standard reference.  This reads the collocation output of a run, which
carries the network prediction at every collocation point, interpolates
onto the two centrelines, and plots both against the reference.

The comparison is qualitative on purpose: the point is whether the
solution has the right shape and magnitude, not whether it matches a
finite-difference benchmark to three digits.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ghia et al. (1982), Re = 100
GHIA_U_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
                     0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
                     0.0625, 0.0547, 0.0000])
GHIA_U = np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151,
                   0.00332, -0.13641, -0.20581, -0.21090, -0.15662, -0.10150,
                   -0.06434, -0.04775, -0.04192, -0.03717, 0.00000])
GHIA_V_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                     0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                     0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914,
                   -0.22445, -0.24533, 0.05454, 0.17527, 0.17507, 0.16077,
                   0.12317, 0.10890, 0.10091, 0.09233, 0.00000])


def strip(x, y, q, fixed, along, halfwidth):
    """values of q in a strip about a centreline, sorted along it"""
    m = np.abs(fixed - 0.5) < halfwidth
    if m.sum() < 8:
        return None, None
    o = np.argsort(along[m])
    return along[m][o], q[m][o]


def smooth(t, q, grid, width):
    """a local average, since the collocation points are scattered"""
    out = np.zeros_like(grid)
    for i, g in enumerate(grid):
        w = np.exp(-((t - g)/width)**2)
        out[i] = (w*q).sum()/max(w.sum(), 1e-300)
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "cavity_compare.pdf"
    f = os.path.join(d, "output_set0002.dat")
    if not os.path.exists(f):
        sys.exit("no output_set0002.dat in " + d)

    rows = []
    for line in open(f):
        if line.lstrip().startswith("#"):
            continue
        v = [float(t) for t in line.split()]
        if len(v) >= 5:
            rows.append(v)
    a = np.array(rows)
    x, y = a[:, 0], a[:, 1]
    u, v_, p = a[:, 2], a[:, 3], a[:, 4]

    grid = np.linspace(0.02, 0.98, 60)
    ty, uy = strip(x, y, u, x, y, 0.06)
    tx, vx = strip(x, y, v_, y, x, 0.06)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))

    if ty is not None:
        ax[0].plot(smooth(ty, uy, grid, 0.03), grid, "b-", lw=1.6,
                   label="PINN")
    ax[0].plot(GHIA_U, GHIA_U_Y, "ro", ms=5, label="Ghia et al. (1982)")
    ax[0].set_xlabel("$u$");  ax[0].set_ylabel("$y$")
    ax[0].set_title("$u$ on $x = 0.5$", fontsize=10)

    if tx is not None:
        ax[1].plot(grid, smooth(tx, vx, grid, 0.03), "b-", lw=1.6,
                   label="PINN")
    ax[1].plot(GHIA_V_X, GHIA_V, "ro", ms=5, label="Ghia et al. (1982)")
    ax[1].set_xlabel("$x$");  ax[1].set_ylabel("$v$")
    ax[1].set_title("$v$ on $y = 0.5$", fontsize=10)

    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8, frameon=False)
    fig.suptitle("Lid-driven cavity, Re = 100", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)

    # a number to go with the picture
    if ty is not None:
        pred = np.interp(GHIA_U_Y, grid, smooth(ty, uy, grid, 0.03))
        print("  u on the vertical centreline: rms difference from Ghia "
              "%.4f  (peak |u| = %.3f)"
              % (np.sqrt(((pred-GHIA_U)**2).mean()), np.abs(GHIA_U).max()))
    if tx is not None:
        pred = np.interp(GHIA_V_X, grid, smooth(tx, vx, grid, 0.03))
        print("  v on the horizontal centreline: rms difference from Ghia "
              "%.4f  (peak |v| = %.3f)"
              % (np.sqrt(((pred-GHIA_V)**2).mean()), np.abs(GHIA_V).max()))


if __name__ == "__main__":
    main()
