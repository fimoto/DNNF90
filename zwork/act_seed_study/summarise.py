#!/usr/bin/env python3
"""Summarise a paired activation study written by run_study.sh.

    python3 summarise.py [activations ...]

Three quantities per run, all over the carried slots of the case:

    R2 mean    averaged over the slots
    R2 worst   the least well fitted slot, which is where an activation
               with growing high derivatives gives way first
    cost       the final training loss

The comparison is paired: the runs of one seed differ only in the
activation, so the seed can be differenced away.  A sign test over the
seeds is reported against the first activation named, which is taken as
the baseline.
"""
import glob
import math
import os
import re
import statistics
import sys


def slots(d):
    deg = {}
    for line in open(os.path.join(d, "hod_alpha_order.dat")):
        if line.startswith("#"):
            continue
        v = [int(t) for t in line.split()]
        deg[v[0]-1] = v[1]
    return deg


def read(d):
    """R2 per slot and the final loss, or None if the run is incomplete"""
    f = os.path.join(d, "output_hod_set0001.dat")
    h = os.path.join(d, "history_ep0000000.dat")
    if not (os.path.exists(f) and os.path.exists(h)):
        return None
    na = len(slots(d))
    # the input width is what is left once the two blocks are removed
    tok = []
    for line in open(f):
        if line.lstrip().startswith("#"):
            continue
        tok += [float(t) for t in line.split()]
    d0 = None
    for cand in range(1, 64):
        if len(tok) % (cand + 2*na) == 0:
            d0 = cand
            break
    if d0 is None:
        return None
    rec = d0 + 2*na
    rows = [tok[i*rec:(i+1)*rec] for i in range(len(tok)//rec)]
    r2 = []
    for c in range(na):
        p = [r[d0+c] for r in rows]
        e = [r[d0+na+c] for r in rows]
        m = sum(e)/len(e)
        var = sum((y-m)**2 for y in e)/len(e)
        if var <= 0:
            continue
        mse = sum((a-b)**2 for a, b in zip(p, e))/len(e)
        r2.append(1 - mse/var)
    loss = float([l.split()[2] for l in open(h)
                  if not l.startswith("#")][-1].replace("E", "e"))
    return sum(r2)/len(r2), min(r2), loss


def main():
    acts = sys.argv[1:] or ["TANH", "SIN"]
    base = acts[0]
    seeds = sorted({re.match(r"run_(\d+)_", os.path.basename(p)).group(1)
                    for p in glob.glob("run_*_*")
                    if re.match(r"run_(\d+)_", os.path.basename(p))})
    data = {}
    for s in seeds:
        row = {}
        for a in acts:
            r = read("run_%s_%s" % (s, a))
            if r:
                row[a] = r
        if len(row) == len(acts):
            data[s] = row
    full = sorted(data)
    if not full:
        sys.exit("no complete seeds found; run run_study.sh first")

    print("  paired activation study, %d complete seeds" % len(full))
    print("  %-8s" % "seed" + "".join("%11s" % a for a in acts)
          + "   (R2 worst slot)")
    for s in full:
        print("  %-8s" % s + "".join("%11.4f" % data[s][a][1] for a in acts))

    print()
    print("  %-9s %10s %10s %10s %12s" %
          ("act", "R2 mean", "R2 worst", "R2 worst", "cost"))
    print("  %-9s %10s %10s %10s %12s" %
          ("", "median", "median", "min", "median"))

    def med(v):
        # The ordinary median: for an even sample the mean of the two
        # middle values, not the upper-middle order statistic.
        return statistics.median(v)

    with open("per_seed.csv", "w") as fh:
        fh.write("seed,activation,r2_mean,r2_worst,cost\n")
        for s in full:
            for a in acts:
                v = data[s][a]
                fh.write("%s,%s,%.6g,%.6g,%.6g\n" % (s, a, v[0], v[1], v[2]))
    print("  per-seed values written to per_seed.csv")

    for a in acts:
        mn = [data[s][a][0] for s in full]
        wr = [data[s][a][1] for s in full]
        co = [data[s][a][2] for s in full]
        print("  %-9s %10.4f %10.4f %10.4f %12.3e"
              % (a, med(mn), med(wr), min(wr), med(co)))

    print()
    n = len(full)
    for a in acts[1:]:
        w = sum(1 for s in full if data[s][a][2] < data[s][base][2])
        p = sum(math.comb(n, i) for i in range(w, n+1)) / 2**n
        print("  %s beats %s on cost %d/%d, sign test p = %.4f"
              % (a, base, w, n, p))
        bw = [data[s][base][1] for s in full]
        aw = [data[s][a][1] for s in full]
        print("     worst-slot ranges: %s [%.4f, %.4f]  %s [%.4f, %.4f]  disjoint %s"
              % (base, min(bw), max(bw), a, min(aw), max(aw),
                 min(aw) > max(bw)))


if __name__ == "__main__":
    main()
