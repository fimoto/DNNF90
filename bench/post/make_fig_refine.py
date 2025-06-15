#!/usr/bin/env python3
"""Figure for the refinement experiment: what a force field knows.

    python3 bench/post/make_fig_refine.py <run root> [out.pdf]

Reads the three arms of zwork/morse_refine, which the README of that
directory explains how to produce, and draws the relative error of each
derivative order.

The line at one is the point of the figure.  A relative error of one
means the model does no better on that order than predicting the mean of
the target, so anything at or above it carries no information.  The
forces-only fit sits below it for the forces and the harmonic constants
and crosses it at the quartic order; the refinement pulls every order
down by more than an order of magnitude; and the control, which runs the
same extra epochs without the high-order targets, does not move.
"""
import collections
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def layout(d):
    deg = {}
    with open(os.path.join(d, "hod_alpha_order.dat")) as fh:
        for l in fh:
            if l.startswith("#"):
                continue
            v = [int(t) for t in l.split()]
            deg[v[0]-1] = v[1]
    na = len(deg)
    tok = []
    with open(os.path.join(d, "output_hod_set0001.dat")) as fh:
        for l in fh:
            if l.lstrip().startswith("#"):
                continue
            tok += [float(t) for t in l.split()]
    d0 = None
    for cand in range(1, 64):
        if len(tok) % (cand + 2*na) == 0:
            d0 = cand
            break
    rec = d0 + 2*na
    return deg, d0, na, [tok[i*rec:(i+1)*rec] for i in range(len(tok)//rec)]


def per_order(d):
    deg, d0, na, rows = layout(d)
    acc = collections.defaultdict(lambda: [0.0, 0.0])
    for c in range(na):
        a = acc[deg[c]]
        for r in rows:
            pr, tg = r[d0+c], r[d0+na+c]
            a[0] += (pr-tg)**2
            a[1] += tg*tg
    return {p: math.sqrt(v[0]/v[1]) for p, v in acc.items() if v[1] > 0}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    out = sys.argv[2] if len(sys.argv) > 2 else "fig_refine.pdf"

    arms = [("stage1", "trained on forces alone", "o", "#444444", "-"),
            ("control", "+4000 epochs, forces only", "s", "#888888", "--"),
            ("stage2", "+4000 epochs, orders 2-4 added", "D", "#1f4e79", "-")]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    orders = [1, 2, 3, 4]
    labels = ["forces\n(1st)", "harmonic\n(2nd)", "cubic\n(3rd)", "quartic\n(4th)"]

    for sub, lab, mk, col, ls in arms:
        d = os.path.join(root, sub)
        if not os.path.exists(os.path.join(d, "output_hod_set0001.dat")):
            print("missing:", d)
            continue
        po = per_order(d)
        ax.plot(orders, [po[p] for p in orders], marker=mk, color=col,
                linestyle=ls, label=lab, markersize=6, linewidth=1.6)

    ax.axhline(1.0, color="#b03030", linewidth=1.0, linestyle=":")
    ax.text(1.05, 1.12, "no information", color="#b03030", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(orders)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"relative error  $\|\Phi_{\rm pred}-\Phi\|_p/\|\Phi\|_p$",
                  fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, which="both", axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
