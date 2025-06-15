#!/usr/bin/env python3
"""Largest value of |sigma^(q)(a)| for each activation the engine carries.

    python3 tools/sigma_growth.py [--qmax 8] [--range 6] [--latex]

This is the measurement behind the activation table of the paper.  The
engine carries sigma^(q) up to q = K+1, so an activation whose derivative
sequence grows makes the deep slots large and badly conditioned, and the
growth is a property of the function rather than of the training.

The derivatives are taken with mpmath at fifty digits, which matters:
differentiating tanh eight times in double precision on a grid loses
enough to change the last figure.  If mpmath is not installed the script
says so rather than reporting numbers it cannot support.

The Bessel entries are the ones the paper leans on.  Every derivative of
J_nu is a binomial combination

    J_nu^(q)(a) = ((-1)^q / 2^q) sum_k (-1)^k C(q,k) J_(nu+q-2k)(a),

and |J_n(a)| <= 1 for every order and every real argument, so the whole
sequence is bounded by one; the measured maxima fall with q.  For sin the
bound is exact at one, since sigma^(q)(a) = sin(a + q pi/2).
"""
import argparse
import sys

try:
    from mpmath import mp, mpf, tanh, sin, besselj, erf, diff
except ImportError:
    sys.exit("this script needs mpmath: pip install mpmath")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qmax", type=int, default=8)
    ap.add_argument("--range", type=float, default=6.0)
    ap.add_argument("--points", type=int, default=1201)
    ap.add_argument("--dps", type=int, default=30)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    mp.dps = args.dps
    A = mpf(args.range)

    acts = [
        ("TANH",    r"$\tanh$",          lambda a: tanh(a)),
        ("ERF",     r"$\mathrm{erf}$",   lambda a: erf(a)),
        ("SIN",     r"$\sin$",           lambda a: sin(a)),
        ("BESSEL",  r"$J_0$",            lambda a: besselj(0, a)),
        ("BESSEL1", r"$J_1$",            lambda a: besselj(1, a)),
    ]

    rows = []
    for name, tex, f in acts:
        vals = []
        for q in range(args.qmax + 1):
            m = mpf(0)
            for i in range(args.points):
                a = -A + 2*A*i/(args.points - 1)
                m = max(m, abs(diff(f, a, q)))
            vals.append(float(m))
        rows.append((name, tex, vals))

    if args.latex:
        cols = [2, 4, 6, 8]
        cols = [q for q in cols if q <= args.qmax]
        print(r"\begin{tabular}{@{}l" + "r"*len(cols) + "@{}}")
        print(r"\toprule")
        print(r"$\sigma$ & " + " & ".join("$q=%d$" % q for q in cols) + r" \\")
        print(r"\midrule")
        for _, tex, v in rows:
            print("%s & " % tex
                  + " & ".join(("%.3g" % v[q]) for q in cols) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print("  max |sigma^(q)(a)| over a in [-%g, %g], %d digits"
              % (args.range, args.range, args.dps))
        print("  %-9s" % "act" + "".join("%10d" % q
                                         for q in range(args.qmax + 1)))
        for name, _, v in rows:
            print("  %-9s" % name + "".join("%10.4g" % x for x in v))


if __name__ == "__main__":
    main()
