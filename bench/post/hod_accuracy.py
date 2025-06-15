#!/usr/bin/env python3
"""Order-resolved accuracy of a high-order fit.

    python3 bench/post/hod_accuracy.py <run directory> [...]

A single coefficient of determination over the carried slots is not
wrong, it is uninformative: it mixes derivative orders whose targets are
all about the same size but whose predictions are not.  On the shipped
K = 7 case the targets have root mean square 0.04 to 0.07 at every order,
while the predictions of a tanh network reach fifteen times the target at
order seven.  One number cannot show that; resolving by order can.

Two quantities are reported for each order p, both dimensionless and
both taken over the slots of that order and over all points:

    rel(p) = || T_pred - y ||_p / || y ||_p     accuracy
    amp(p) = || T_pred    ||_p / || y ||_p      blow-up detector

rel near or above one means the order carries no information.  amp near
one means the model has the right size there; amp much larger than one
means it is diverging at that order, which is the failure an activation
with bounded high derivatives is meant to prevent.

A single figure for ranking runs is the relative error in the norm the
loss actually minimizes,

    ||T - y||_lambda / ||y||_lambda,   weights lambda_p from input_nn.dat,

which is scale free and weights each order the way the training did.

Reads output_hod_set0001.dat, hod_alpha_order.dat and input_nn.dat from
each directory, so it needs no rerun.
"""

import collections
import math
import os
import re
import sys


def read_orders(d):
    """slot index (0 based) -> order of its multi-index"""
    deg = {}
    with open(os.path.join(d, "hod_alpha_order.dat")) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            v = [int(t) for t in line.split()]
            deg[v[0]-1] = v[1]
    return deg


def read_lambda(d, kmax):
    """the K+1 weights that follow Hod_K in the input file"""
    txt = open(os.path.join(d, "input_nn.dat")).read()
    m = re.search(r"^\s*Hod_K[^\n]*\n((?:[^\n]*\n){%d})" % (kmax+1), txt,
                  re.M | re.I)
    if not m:
        return [1.0]*(kmax+1)
    out = []
    for line in m.group(1).strip().split("\n"):
        tok = line.split()
        if not tok:
            continue
        out.append(float(tok[0].replace("d", "e").replace("D", "E")))
    return out if len(out) == kmax+1 else [1.0]*(kmax+1)


def read_rows(d, d0, na):
    """one logical record per point: x(1:d0), predictions, targets"""
    rec = d0 + 2*na
    tok = []
    with open(os.path.join(d, "output_hod_set0001.dat")) as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                continue
            tok += [float(t) for t in line.split()]
    n = len(tok)//rec
    return [tok[i*rec:(i+1)*rec] for i in range(n)]


def analyse(d):
    deg = read_orders(d)
    na = len(deg)
    kmax = max(deg.values())
    # the input width is whatever is left once the two blocks are removed
    ntok = 0
    with open(os.path.join(d, "output_hod_set0001.dat")) as fh:
        for line in fh:
            if not line.lstrip().startswith("#"):
                ntok += len(line.split())
    # try the plausible widths and take the one that divides evenly
    d0 = None
    for cand in range(1, 64):
        if ntok % (cand + 2*na) == 0:
            d0 = cand
            break
    if d0 is None:
        sys.exit("%s: cannot infer the input width" % d)

    rows = read_rows(d, d0, na)
    lam = read_lambda(d, kmax)

    acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])   # err2 tgt2 prd2
    wnum = wden = 0.0
    for c in range(na):
        p = deg[c]
        a = acc[p]
        w = lam[p] if p < len(lam) else 0.0
        for r in rows:
            pr, tg = r[d0+c], r[d0+na+c]
            e2 = (pr-tg)**2
            a[0] += e2
            a[1] += tg*tg
            a[2] += pr*pr
            wnum += w*e2
            wden += w*tg*tg
    per = {}
    for p, a in sorted(acc.items()):
        per[p] = (math.sqrt(a[0]/a[1]) if a[1] > 0 else float("nan"),
                  math.sqrt(a[2]/a[1]) if a[1] > 0 else float("nan"),
                  len([c for c in deg if deg[c] == p]))
    overall = math.sqrt(wnum/wden) if wden > 0 else float("nan")
    return d0, na, per, overall


def main():
    dirs = sys.argv[1:] or ["."]
    for d in dirs:
        if not os.path.exists(os.path.join(d, "output_hod_set0001.dat")):
            print("%s: no output_hod_set0001.dat, skipped" % d)
            continue
        d0, na, per, overall = analyse(d)
        print("%s   D0 = %d, %d carried slots" % (d, d0, na))
        print("   order  slots   rel = |T-y|/|y|   amp = |T|/|y|")
        for p in sorted(per):
            rel, amp, ns = per[p]
            flag = ""
            if rel >= 1.0:
                flag = "   <- order lost"
            elif amp > 2.0:
                flag = "   <- diverging"
            print("   %5d %6d %16.4f %15.3f%s" % (p, ns, rel, amp, flag))
        print("   relative error in the loss norm: %.5f" % overall)
        print()


if __name__ == "__main__":
    main()
