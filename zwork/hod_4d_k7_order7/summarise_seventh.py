#!/usr/bin/env python3
"""Summarise the seventh-order seed study.

Reads every run_<seed>_<activation>/ left by run_seed_study.sh and
writes per_seed.csv and the table the paper reports: the median loss
over seeds, its range, and the relative error of the seventh
derivatives, ||T-y||_7 / ||y||_7, over the 120 multi-indices of order
exactly seven and all points.  A value of one is the error predicting
zero would give.  A run that did not reach the epoch count the case
declares is left out rather than compared against complete ones.
"""
import glob
import os
import re
import statistics
import sys


def order_of(path):
    deg = []
    for line in open(path):
        f = line.split()
        if f and f[0].lstrip("-").isdigit():
            deg.append(int(f[1]))
    return deg


def best_loss(run, want):
    path = os.path.join(run, "history_ep0000000.dat")
    if not os.path.exists(path):
        return None
    ep, cost = [], []
    for line in open(path):
        if line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) >= 3:
            try:
                ep.append(float(f[0])); cost.append(float(f[2]))
            except ValueError:
                pass
    if not cost or (want is not None and ep[-1] < want):
        return None
    return min(cost)


def rel_error(run, deg):
    path = os.path.join(run, "output_hod_set0001.dat")
    if not os.path.exists(path):
        return None
    na = len(deg)
    num = den = 0.0
    for line in open(path):
        if line.lstrip().startswith("#"):
            continue
        v = [float(t) for t in line.split()]
        if len(v) < 2 * na:
            continue
        pred, targ = v[-2 * na:-na], v[-na:]
        for i, d in enumerate(deg):
            if d == 7:
                num += (pred[i] - targ[i]) ** 2
                den += targ[i] ** 2
    return (num / den) ** 0.5 if den > 0 else None


def main():
    acts = sys.argv[1:] or ["TANH", "SIN", "BESSEL", "BESSEL1"]
    runs = sorted(glob.glob("run_*_[A-Z]*"))
    if not runs:
        print("no run_* directories; run run_seed_study.sh first")
        sys.exit(1)
    want = None
    for line in open("input_nn.dat"):
        if line.strip().startswith("Epoch"):
            want = float(line.split()[1]); break
    deg = order_of(os.path.join(runs[0], "hod_alpha_order.dat"))
    data = {}
    for r in runs:
        m = re.match(r"run_(\d+)_([A-Z0-9]+)$", r)
        if not m:
            continue
        s, a = m.group(1), m.group(2)
        L, e = best_loss(r, want), rel_error(r, deg)
        if L is None or e is None:
            continue
        data.setdefault(s, {})[a] = (L, e)
    # only seeds where every activation completed enter the comparison,
    # so every column is over the same population
    full = [s for s in sorted(data) if all(a in data[s] for a in acts)]
    with open("per_seed.csv", "w") as fh:
        fh.write("seed,activation,loss,rel_error_order7\n")
        for s in full:
            for a in acts:
                fh.write("%s,%s,%.6g,%.6g\n" % (s, a, data[s][a][0], data[s][a][1]))
    print("  seventh-order seed study, %d complete seeds" % len(full))
    print("  per-seed values written to per_seed.csv")
    print("  %-9s %9s %20s %9s %11s" %
          ("act", "loss med", "loss range", "rel err", "beats TANH"))
    for a in acts:
        L = [data[s][a][0] for s in full]
        e = [data[s][a][1] for s in full]
        wins = sum(1 for s in full if data[s][a][0] < data[s]["TANH"][0])
        print("  %-9s %9.4g  [%8.4g, %8.4g] %9.3f %7d/%d" %
              (a, statistics.median(L), min(L), max(L),
               statistics.median(e), wins, len(full)))


main()
