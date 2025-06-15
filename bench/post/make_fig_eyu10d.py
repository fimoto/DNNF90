#!/usr/bin/env python3
"""Draw fig_eyu10d.pdf from the eyu10d run.

The other four figures of the paper come from post/make_figs.py; this one
had no script in the repository, so it is reconstructed from the caption:

    Left:   exact solution on the (x1,x2) slice, the remaining eight
            coordinates fixed at 0.5.
    Middle: the trained network on the same slice.
    Right:  parity plot against the exact solution on the interior points
            of the run.

Run it from the directory that holds the eyu10d case, that is from
bench/ or from a staging copy:

    python3 post/make_fig_eyu10d.py            # writes figs/fig_eyu10d.pdf

It reads only eyu10d/nn_weight.dat and eyu10d/colloc.dat, so it does not
retrain anything.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_eval import load_net, forward           # noqa: E402

CASE = "eyu10d"
D0 = 10
FIXED = 0.5          # the eight coordinates held fixed on the slice
NGRID = 120


def u_exact(X):
    """Eq. (15) of E and Yu: u = sum_k x_{2k-1} x_{2k}, which is harmonic."""
    X = np.asarray(X, float)
    return sum(X[:, 2 * k] * X[:, 2 * k + 1] for k in range(D0 // 2))


def main():
    for f in (f"{CASE}/nn_weight.dat", f"{CASE}/colloc.dat"):
        if not os.path.exists(f):
            sys.exit("%s not found: run the case first so that its weights "
                     "and collocation points are present" % f)

    dims, Ws, bs = load_net(f"{CASE}/nn_weight.dat")
    if dims[0] != D0:
        sys.exit("%s has %d inputs, expected %d" % (CASE, dims[0], D0))

    # --- the (x1,x2) slice ---
    g = np.linspace(0.0, 1.0, NGRID)
    X1, X2 = np.meshgrid(g, g)
    P = np.full((X1.size, D0), FIXED)
    P[:, 0] = X1.ravel()
    P[:, 1] = X2.ravel()
    ue = u_exact(P).reshape(X1.shape)
    un = forward(Ws, bs, P).reshape(X1.shape)

    # --- parity on the interior points the run used ---
    Xc = np.loadtxt(f"{CASE}/colloc.dat")[:, :D0]
    ye = u_exact(Xc)
    yn = forward(Ws, bs, Xc)
    rel = np.linalg.norm(yn - ye) / np.linalg.norm(ye)

    fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.5))
    lo, hi = float(min(ue.min(), un.min())), float(max(ue.max(), un.max()))

    for a, Z, t in ((ax[0], ue, "exact"), (ax[1], un, "network")):
        im = a.pcolormesh(X1, X2, Z, shading="auto", vmin=lo, vmax=hi,
                          cmap="viridis", rasterized=True)
        a.set_xlabel("$x_1$")
        a.set_title("%s, $x_{3..10}=%.1f$" % (t, FIXED))
        a.set_aspect("equal")
        fig.colorbar(im, ax=a, fraction=0.046)
    ax[0].set_ylabel("$x_2$")

    ax[2].plot([ye.min(), ye.max()], [ye.min(), ye.max()], "k-", lw=0.8)
    ax[2].plot(ye, yn, ".", ms=2, alpha=0.4, rasterized=True)
    ax[2].set_xlabel("exact $u^*$")
    ax[2].set_ylabel("network $N$")
    ax[2].set_title("%d interior points, rel. $L_2$ = %.2f%%"
                    % (len(ye), 100.0 * rel))

    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/fig_eyu10d.pdf", dpi=300)
    print("figs/fig_eyu10d.pdf written; relative L2 = %.4f on %d points"
          % (rel, len(ye)))


if __name__ == "__main__":
    main()
