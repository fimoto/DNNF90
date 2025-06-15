#!/usr/bin/env python3
"""Compare the shipped results against the numbers the paper reports.

tools/repro_check.py answers "is the input present".  This one answers
"does the number still come out", for every row of
results/reference/manifest.csv that carries an expected value.  It reads
the result files that ship with the distribution; it does not run
anything, so a fresh set of runs must be made first if the point is to
check a rebuild.

    python3 tools/verify_results.py            # check the shipped files
    python3 tools/verify_results.py --list     # show the manifest

Tolerances are relative and generous, because training is chaotic in the
seed and the machine: a row passes when the value is within the stated
factor of the reported one, which catches a changed algorithm rather
than a changed CPU.
"""
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(R, "results", "reference", "manifest.csv")


def final_cost(path):
    """The cost at the last validation event, which is what the
    benchmark table reports (column "final cost")."""
    last = None
    for line in open(path):
        f = line.split()
        if len(f) >= 3 and not line.lstrip().startswith("#"):
            try:
                last = float(f[2])
            except ValueError:
                pass
    return last


def best_cost(path):
    """The cost the run reports as its best, from the case's a.log.

    The paper quotes the best epoch, not the last one: training is not
    monotone and the final validation event can be above the minimum.
    """
    for line in reversed(open(path).readlines()):
        if "best epoch=" in line:
            return float(line.split()[-1].replace("D", "E"))
    return None


def rows():
    for line in open(MANIFEST):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        f = line.split(",")
        if len(f) == 6:
            yield f


def main():
    if "--list" in sys.argv:
        for f in rows():
            print("%-28s %s" % (f[0], f[1]))
        return
    npass = nfail = nskip = 0
    for name, cmd, src, metric, exp, tol in rows():
        path = os.path.join(R, src)
        if not os.path.exists(path):
            print("  MISSING  %-28s %s" % (name, src))
            nfail += 1
            continue
        if not exp:
            print("  present  %-28s %s" % (name, src))
            nskip += 1
            continue
        if metric == "best_cost":
            got = best_cost(path)
        elif metric == "final_cost":
            got = final_cost(path)
        else:
            got = None
        if got is None:
            print("  NO VALUE %-28s %s" % (name, src))
            nfail += 1
            continue
        e, t = float(exp), float(tol)
        ok = abs(got - e) <= t * abs(e)
        print("  %-8s %-28s %s = %.3e (paper %.3e)"
              % ("ok" if ok else "DIFFERS", name, metric, got, e))
        if ok:
            npass += 1
        else:
            nfail += 1
    print("\n  %d compared and agreeing, %d differing, %d present but not "
          "compared" % (npass, nfail, nskip))
    sys.exit(1 if nfail else 0)


main()
