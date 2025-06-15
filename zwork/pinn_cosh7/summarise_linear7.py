#!/usr/bin/env python3
"""Summarise the seed study of the linear seventh-order equation.

Reads every run_<seed>_<activation>/ left by run_seed_study.sh and
writes per_seed.csv and the table the paper reports: the median R^2
against the exact solution over seeds, its range, and how often each
activation beats tanh on the same seed.

R^2 is taken on the collocation set, from output_set0002.dat, whose
last two columns are the model and the exact solution.
"""
import glob
import os
import re
import statistics
import sys


def r2(run):
    path = os.path.join(run, "output_set0002.dat")
    if not os.path.exists(path):
        return None
    pred, exact = [], []
    for line in open(path):
        if line.lstrip().startswith("#"):
            continue
        f = line.split()
        if len(f) >= 4:
            try:
                pred.append(float(f[-2])); exact.append(float(f[-1]))
            except ValueError:
                pass
    if len(exact) < 2:
        return None
    m = sum(exact) / len(exact)
    ss_res = sum((p - e) ** 2 for p, e in zip(pred, exact))
    ss_tot = sum((e - m) ** 2 for e in exact)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else None


def complete(run, want):
    """True if the run reached the epoch count the case declares."""
    path = os.path.join(run, "history_ep0000000.dat")
    if not os.path.exists(path):
        return False
    last = None
    for line in open(path):
        f = line.split()
        if len(f) >= 3 and not line.lstrip().startswith("#"):
            try:
                last = float(f[0])
            except ValueError:
                pass
    return last is not None and (want is None or last >= want)


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
    data = {}
    for r in runs:
        m = re.match(r"run_(\d+)_([A-Z0-9]+)$", r)
        if not m or not complete(r, want):
            continue
        v = r2(r)
        if v is not None:
            data.setdefault(m.group(1), {})[m.group(2)] = v
    full = [s for s in sorted(data) if all(a in data[s] for a in acts)]
    with open("per_seed.csv", "w") as fh:
        fh.write("seed,activation,r2\n")
        for s in full:
            for a in acts:
                fh.write("%s,%s,%.6g\n" % (s, a, data[s][a]))
    print("  linear seventh-order seed study, %d complete seeds" % len(full))
    print("  per-seed values written to per_seed.csv")
    print("  %-9s %9s %22s %11s" % ("act", "R2 median", "R2 range", "beats TANH"))
    for a in acts:
        v = [data[s][a] for s in full]
        wins = sum(1 for s in full if data[s][a] > data[s]["TANH"])
        print("  %-9s %9.3f  [%9.3f, %9.3f] %7d/%d" %
              (a, statistics.median(v), min(v), max(v), wins, len(full)))


main()
