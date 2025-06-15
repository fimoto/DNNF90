#!/usr/bin/env python3
"""The observation-space EnKF analysis of an iPINNER-style outer loop
(Lu, Mou & Lin, arXiv:2506.00731), adapted to this code base: the
Pareto-spread NSGA-III ensemble of the paper is replaced by cheap
full-batch L-BFGS members that differ in Rand_seed (and, if desired,
in the data/residual Loss_term weights, which is the same trade-off
axis), and the EnKF runs where the paper runs it -- on the PREDICTED
FIELD VALUES at the data points, never on the network weights.  That
placement is what makes it immune to the sequential-scalarization
failure the weight-space filters of this code base were measured to
have on coupled systems (see zwork/FLOW_CASES.md).

One outer iteration:

  1. train M members on the noisy data (different Rand_seed), e.g.
       for s in 101 202 303 404; do ... input_cold.dat ... ; done
  2. run this script: it reads each member's output_set<data>.dat,
     forms the fully localized perturbed-observation EnKF analysis at
     every data point, and writes data_refined.dat into each member
     directory;
  3. retrain each member from its weights (Restart 1) with
     data_refined.dat as its data file;
  4. repeat from 2 if wanted.

With a handful of members the sample covariance is rank M-1, so the
textbook joint update is unusable; the per-point, per-component
scalar analysis used here is the honest small-ensemble form, and
--inflation covers the underdispersion of a biased prior (wrong or
missing physics), which is the regime this loop pays in.  Measured on
the five-component EHD case (10% observation noise, 3-4 members):
with correct physics the physics loss already filters the noise and
one iteration buys little (R2 +0.0005..0.0013 per component); with a
2x-wrong viscosity one inflated iteration improves every component,
most where the bias hurt most (p: R2 0.9779 -> 0.9805).

Usage:
  ipinner_enkf.py --members m101 m202 ... --data data_noisy.dat \
                  --sigma s1 s2 ... [--set 1] [--inflation 1.0]
"""
import argparse, os, sys
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True,
                    help='member run directories')
    ap.add_argument('--data', required=True,
                    help='the (noisy) observation file: coords + fields')
    ap.add_argument('--sigma', nargs='+', type=float, required=True,
                    help='observation noise per component')
    ap.add_argument('--ncoord', type=int, default=2,
                    help='number of coordinate columns (default 2)')
    ap.add_argument('--set', type=int, default=1,
                    help='which output_setNNNN.dat is the data set')
    ap.add_argument('--inflation', type=float, default=1.0,
                    help='multiplier on the ensemble SPREAD before the '
                         'gain; >1 for a biased (wrong-physics) prior')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed if a.seed else None)
    y = np.loadtxt(a.data)
    nc = a.ncoord
    m = y.shape[1] - nc
    sig = np.asarray(a.sigma, float)
    if sig.size != m:
        sys.exit(f"--sigma needs {m} values for {m} field columns")

    fname = f"output_set{a.set:04d}.dat"
    u = []
    for d in a.members:
        p = np.loadtxt(os.path.join(d, fname))
        if p.shape[0] != y.shape[0]:
            sys.exit(f"{d}/{fname}: {p.shape[0]} rows, data has {y.shape[0]}")
        u.append(p[:, nc:nc+m])          # predictions follow the coords
    u = np.array(u)
    if u.shape[0] < 2:
        sys.exit("need at least 2 members for an ensemble spread")

    s2 = u.var(axis=0, ddof=1) * a.inflation**2
    gain = s2/(s2 + sig[None, :]**2)
    print("mean Kalman gain per component:",
          " ".join(f"{g:.2f}" for g in gain.mean(axis=0)))

    for i, d in enumerate(a.members):
        ypert = y[:, nc:] + rng.normal(0, 1, size=u[i].shape)*sig[None, :]
        ua = u[i] + gain*(ypert - u[i])
        out = np.hstack([y[:, :nc], ua])
        np.savetxt(os.path.join(d, "data_refined.dat"), out, fmt="%22.14e")
        print(f"wrote {d}/data_refined.dat")

if __name__ == "__main__":
    main()
