#!/usr/bin/env python3
"""Figure for the coupled-system case: the Taylor-Green vortex.

    python3 bench/post/make_fig_taylorgreen.py <run dir> [out.pdf]

Reads output_set0002.dat, the collocation points, whose last columns
carry the exact solution the case was generated from.  Three field
components share one network, and the momentum equations couple them
through u u_x + v u_y, so this is the figure that shows the System block
doing what it exists for.

The top row shows the exact solution at the latest time in the data, the
middle row the network, and the bottom row their difference on the same
colour scale as the field itself, so that a small residual error is
visibly small rather than amplified by its own scale.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            v = [float(t) for t in line.split()]
            if len(v) >= 9:
                rows.append(v)
    return np.array(rows)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "fig_taylorgreen.pdf"
    f = os.path.join(d, "output_set0002.dat")
    if not os.path.exists(f):
        sys.exit("no output_set0002.dat in " + d)

    a = load(f)
    x, y, t = a[:, 0], a[:, 1], a[:, 2]
    pred, exact = a[:, 3:6], a[:, 6:9]

    # one time slice: the points closest to the latest time present
    tsel = t.max()
    m = np.abs(t - tsel) < 0.12 * (t.max() - t.min() + 1e-30)
    if m.sum() < 40:
        m = np.ones_like(t, dtype=bool)

    names = [r"$u$", r"$v$", r"$p$"]
    fig, ax = plt.subplots(3, 3, figsize=(7.4, 6.4))

    for c in range(3):
        e, p = exact[m, c], pred[m, c]
        lim = max(abs(e).max(), 1e-12)
        common = dict(cmap="RdBu_r", vmin=-lim, vmax=lim, s=14,
                      edgecolors="none")
        ax[0, c].scatter(x[m], y[m], c=e, **common)
        ax[1, c].scatter(x[m], y[m], c=p, **common)
        im = ax[2, c].scatter(x[m], y[m], c=p-e, **common)

        # the coefficient of determination over all points, not the slice
        ee, pp = exact[:, c], pred[:, c]
        r2 = 1 - ((pp-ee)**2).mean()/ee.var()
        ax[0, c].set_title(names[c], fontsize=11)
        ax[2, c].set_xlabel(r"$R^2 = %.3f$" % r2, fontsize=9)
        fig.colorbar(im, ax=ax[:, c], shrink=0.55, pad=0.02)

    for r, lab in enumerate(["exact", "network", "difference"]):
        ax[r, 0].set_ylabel(lab, fontsize=10)
    for a_ in ax.ravel():
        a_.set_xticks([]); a_.set_yticks([]); a_.set_aspect("equal")

    fig.suptitle("Taylor-Green vortex at $t = %.2f$, three components "
                 "from one network" % tsel, fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
