#!/usr/bin/env python3
"""Aggregate the seventh-order activation runs into the paper's table.

Reads run_<seed>_<activation>/a.log, writes per_seed.csv, and prints the
median loss, the range over seeds, the relative error in the seventh
derivatives and the sign-test count against tanh -- the four columns of
the table in the manuscript.
"""
import glob
import os
import re
import statistics
import sys

acts = sys.argv[1:] or ["TANH", "SIN", "BESSEL", "BESSEL1"]
here = os.path.dirname(os.path.abspath(__file__))


def read_run(d):
    """best loss and the relative seventh-derivative error of one run."""
    log = os.path.join(d, "a.log")
    if not os.path.exists(log):
        return None
    txt = open(log).read()
    m = re.findall(r"best epoch=\s*\d+\s+([0-9.ED+-]+)", txt)
    if not m:
        return None
    loss = float(m[-1].replace("D", "E"))
    return loss, rel_error_order7(d)


def rel_error_order7(d):
    """||T - y||/||y|| over the multi-indices of order exactly seven.

    output_hod_set0001.dat holds, per point, x(1:D0) then the carried
    predictions and then the targets in the order of hod_alpha_order.dat.
    """
    ao = os.path.join(d, "hod_alpha_order.dat")
    fo = os.path.join(d, "output_hod_set0001.dat")
    if not (os.path.exists(ao) and os.path.exists(fo)):
        return float("nan")
    order, d0 = [], None
    for line in open(ao):
        f = line.split()
        if not f or not f[0].isdigit():
            continue
        order.append(int(f[1]))
        d0 = len(f) - 2
    na = len(order)
    idx = [i for i, o in enumerate(order) if o == 7]
    if not idx or d0 is None:
        return float("nan")
    num = den = 0.0
    for line in open(fo):
        if line.lstrip().startswith("#"):
            continue
        v = [float(t) for t in line.split()]
        if len(v) < d0 + 2 * na:
            continue
        pred = v[d0:d0 + na]
        targ = v[d0 + na:d0 + 2 * na]
        for i in idx:
            num += (pred[i] - targ[i]) ** 2
            den += targ[i] ** 2
    return (num / den) ** 0.5 if den > 0 else float("nan")


data = {}
for a in acts:
    rows = []
    for d in sorted(glob.glob(os.path.join(here, "run_*_%s" % a))):
        v = read_run(d)
        if v:
            seed = os.path.basename(d).split("_")[1]
            rows.append((seed, v[0], v[1]))
    data[a] = rows

with open(os.path.join(here, "per_seed.csv"), "w") as fh:
    fh.write("seed,activation,loss,rel_error_order7\n")
    for a in acts:
        for seed, loss, rel in data[a]:
            fh.write("%s,%s,%.6g,%.6g\n" % (seed, a, loss, rel))

print("  seventh-order activation study, %d seeds" % len(data[acts[0]]))
print("  %-9s %10s %22s %12s %10s" % ("act", "loss med", "loss range",
                                      "rel err med", "beats tanh"))
base = {s: l for s, l, _ in data.get("TANH", [])}
for a in acts:
    rows = data[a]
    if not rows:
        continue
    ls = [l for _, l, _ in rows]
    rs = [r for _, _, r in rows if r == r]
    wins = sum(1 for s, l, _ in rows if s in base and l < base[s])
    print("  %-9s %10.4f  [%8.4f, %8.4f] %12s %6d/%d"
          % (a, statistics.median(ls), min(ls), max(ls),
             ("%.4f" % statistics.median(rs)) if rs else "-",
             wins, len(rows)))
print("  per-seed values written to per_seed.csv")
