#!/usr/bin/env python3
"""Draw fig_forcefield.pdf from the force-field demonstration.

The demonstration (tools/example_hod_ff.f90) trains two networks on the
same Morse chain from the same start: one on energies and forces, which is
what a descriptor pipeline can supply, and one that also sees the analytic
Hessians.  It prints the force constants and the phonon frequencies of the
equilibrium chain against the analytic Morse result.

    ./build/hod_ff_example.out > ff.log
    python3 bench/post/make_fig_forcefield.py ff.log

Left:   phonon frequencies, both models against the analytic values.
Right:  relative error of each frequency, which is the quantity the
        Hessian term improves.
"""

import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path):
    """Pull the frequency table and the summary numbers out of the log."""
    txt = open(path).read()
    m = re.search(r"phonon frequencies \(analytic \| E\+F \| E\+F\+H\):\n(.*?)\n"
                  r"(?:max phonon error|###|\Z)", txt, re.S)
    if not m:
        sys.exit("%s does not contain the frequency table; run the example "
                 "and keep its standard output" % path)
    rows = []
    for line in m.group(1).strip().split("\n"):
        f = line.split()
        if len(f) == 3:
            rows.append([float(v) for v in f])
    if not rows:
        sys.exit("%s: the frequency table is empty" % path)
    w = np.array(rows)

    def num(pat, default=np.nan):
        mm = re.search(pat, txt)
        return float(mm.group(1)) if mm else default

    return dict(
        w_ref=w[:, 0], w_ef=w[:, 1], w_efh=w[:, 2],
        fc_ef=num(r"E\+F trained\s*:\s*([0-9.E+-]+)"),
        fc_efh=num(r"E\+F\+H trained\s*:\s*([0-9.E+-]+)"),
        ph_ef=num(r"max phonon error: E\+F\s+([0-9.]+)"),
        ph_efh=num(r"E\+F\+H\s+([0-9.]+)\s*%"),
        chk=re.search(r"ALL PASSED", txt) is not None,
    )


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else "ff.log"
    if not os.path.exists(log):
        sys.exit("%s not found; run build/hod_ff_example.out and keep its "
                 "output" % log)
    d = parse(log)
    if not d["chk"]:
        sys.stderr.write("warning: the log does not report ALL PASSED for the "
                         "chain-rule checks\n")

    n = np.arange(1, len(d["w_ref"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax[0].plot(n, d["w_ref"], "k-o", ms=5, lw=1.2, label="analytic Morse")
    ax[0].plot(n, d["w_ef"], "s--", ms=5, lw=1.0,
               label="trained on $E$, $F$")
    ax[0].plot(n, d["w_efh"], "^:", ms=6, lw=1.0,
               label="trained on $E$, $F$, $H$")
    ax[0].set_xlabel("mode")
    ax[0].set_ylabel(r"frequency $\omega$")
    ax[0].set_xticks(n)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_title("phonons of the equilibrium chain", fontsize=10)

    e_ef = 100.0 * np.abs(d["w_ef"] - d["w_ref"]) / d["w_ref"]
    e_efh = 100.0 * np.abs(d["w_efh"] - d["w_ref"]) / d["w_ref"]
    wdt = 0.36
    ax[1].bar(n - wdt / 2, e_ef, wdt, label="$E$, $F$")
    ax[1].bar(n + wdt / 2, e_efh, wdt, label="$E$, $F$, $H$")
    ax[1].set_xlabel("mode")
    ax[1].set_ylabel("relative error [%]")
    ax[1].set_xticks(n)
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("max %.2f%% against %.2f%%" % (d["ph_ef"], d["ph_efh"]),
                    fontsize=10)

    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig("figs/fig_forcefield.pdf")
    print("figs/fig_forcefield.pdf written")
    print("  force constants, max relative error: E+F %.4f, E+F+H %.4f"
          % (d["fc_ef"], d["fc_efh"]))
    print("  phonons, max relative error:         E+F %.2f%%, E+F+H %.2f%%"
          % (d["ph_ef"], d["ph_efh"]))


if __name__ == "__main__":
    main()
