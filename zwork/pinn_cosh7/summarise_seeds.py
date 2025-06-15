#!/usr/bin/env python3
"""Collect the paired-seed runs of run_seeds.sh into per_seed.csv.

R^2 is computed against the exact solution carried in the last column of
output_set0002.dat, at the collocation points, for every run directory
run_SEED_ACTIVATION that exists.
"""
import glob
import os
import statistics
import sys

import numpy as np


def r2(path):
    d = np.loadtxt(path)
    pred, exact = d[:, -2], d[:, -1]
    ss_res = float(((pred - exact) ** 2).sum())
    ss_tot = float(((exact - exact.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


acts = sys.argv[1:] or ["TANH", "SIN", "BESSEL", "BESSEL1"]
rows = []
for d in sorted(glob.glob("run_*_*")):
    _, seed, act = d.split("_", 2)
    f = os.path.join(d, "output_set0002.dat")
    if os.path.exists(f):
        rows.append((seed, act, r2(f)))

with open("per_seed.csv", "w") as fh:
    fh.write("seed,activation,r2\n")
    for s, a, v in rows:
        fh.write("%s,%s,%.6g\n" % (s, a, v))

print("  %-9s %8s %18s %s" % ("act", "R2 median", "R2 range", "beats TANH"))
base = {s: v for s, a, v in rows if a == "TANH"}
for a in acts:
    v = [x for _, aa, x in rows if aa == a]
    if not v:
        continue
    wins = sum(1 for s, aa, x in rows if aa == a and s in base and x > base[s])
    tag = "---" if a == "TANH" else "%d/%d" % (wins, len(v))
    print("  %-9s %8.3f   [%6.3f, %6.3f]  %s"
          % (a, statistics.median(v), min(v), max(v), tag))
print("  per-seed values written to per_seed.csv")
