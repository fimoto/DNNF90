#!/usr/bin/env python3
"""Generate hod_alpha_order.dat without running a training case.

The column order of every high-order target file is the order in which the
library enumerates the carried multi-indices.  That order is only written
out after a run, which makes preparing a HOD_DATA file circular: the file
needs the column count and the column order, and both come from the run
that is supposed to read the file.  This script breaks the circle by
reproducing the enumeration exactly.

    # dense set, |alpha| <= K
    tools/alpha_order.py --d0 4 --k 3

    # downward closure of the seeds in a file, same format as
    # the file named by Hod_alpha_file
    tools/alpha_order.py --d0 4 --seeds zwork/hod_4d_k7_active/alpha_seeds.dat

    # count only
    tools/alpha_order.py --d0 10 --k 3 --count

The output is byte-identical to the hod_alpha_order.dat the library
writes, so it can be diffed against a real run to confirm agreement.

Ordering.  Multi-indices are enumerated by total order first, and within
one order by the combinations-with-repetition of axis indices taken in
nondecreasing order.  Selecting axis i once contributes one to alpha_i,
so the tuple (1,1) at order two gives alpha = (2,0,...) and the tuple
(1,2) gives (1,1,0,...).  For a closure the same enumeration runs and the
indices outside the closure are dropped, so a closure keeps the relative
order of the dense set.
"""

import argparse
import itertools
import sys


def dense(d0, k):
    """Every multi-index with total order at most k, in library order."""
    out = []
    for p in range(0, k + 1):
        if p == 0:
            out.append((0,) * d0)
            continue
        for slots in itertools.combinations_with_replacement(range(d0), p):
            a = [0] * d0
            for s in slots:
                a[s] += 1
            out.append(tuple(a))
    return out


def read_seeds(path, d0):
    vals = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals.extend(int(v) for v in line.split())
    n = vals[0]
    rest = vals[1:]
    if len(rest) < n * d0:
        sys.exit("%s: %d seeds of %d components need %d values, found %d"
                 % (path, n, d0, n * d0, len(rest)))
    return [tuple(rest[i * d0:(i + 1) * d0]) for i in range(n)]


def leq(b, a):
    """Componentwise b <= a."""
    return all(x <= y for x, y in zip(b, a))


def closure(d0, seeds):
    """Downward closure of the seeds, in library order.

    The closure is the union of the boxes {b : b <= a} over the seeds, so
    its size is bounded by the sum of prod(a_i + 1) and equals that sum
    only when the boxes are disjoint.  The library carries this set
    because the recursions reference, for a target alpha, the indices
    alpha - beta with beta <= alpha: an index whose sub-indices are absent
    cannot be computed.
    """
    k = max(sum(a) for a in seeds)
    return [b for b in dense(d0, k) if any(leq(b, a) for a in seeds)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d0", type=int, required=True,
                    help="number of input variables, the width of layer 1")
    ap.add_argument("--k", type=int, help="maximum order of the dense set")
    ap.add_argument("--seeds", help="seed file, as named by Hod_alpha_file")
    ap.add_argument("--count", action="store_true",
                    help="print only the number of carried multi-indices")
    ap.add_argument("-o", "--out", help="write to this file instead of stdout")
    args = ap.parse_args()

    if (args.k is None) == (args.seeds is None):
        ap.error("give exactly one of --k (dense) or --seeds (closure)")
    if args.d0 < 1:
        ap.error("--d0 must be at least 1")

    if args.seeds:
        seeds = read_seeds(args.seeds, args.d0)
        for a in seeds:
            if any(v < 0 for v in a):
                sys.exit("seed with a negative component: %s" % (a,))
        alist = closure(args.d0, seeds)
    else:
        if args.k < 0:
            ap.error("--k must not be negative")
        alist = dense(args.d0, args.k)

    if args.count:
        print(len(alist))
        return

    lines = ["# column order of high-order derivative targets",
             "# ia  |alpha|  alpha(1:D0)"]
    for ia, a in enumerate(alist, start=1):
        lines.append("%d  %d  %s" % (ia, sum(a), "".join(" %d" % v for v in a)))
    text = "\n".join(lines) + "\n"

    if args.out:
        open(args.out, "w").write(text)
        sys.stderr.write("%d multi-indices written to %s\n" % (len(alist), args.out))
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
