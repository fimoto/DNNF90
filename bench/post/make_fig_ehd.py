"""Figure for the electrohydrodynamic section: the two cold-start
routes and the parity of the five components.

Usage:
    python3 make_fig_ehd.py LBFGS_DIR NGD_DIR [outdir]

LBFGS_DIR holds the finished run of input_cold.dat (history_ep0000000.dat
and output_set0002.dat), NGD_DIR the finished run of input_cold_ngd.dat.
Writes fig_ehd.pdf to outdir (default: .).
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.3,
                     "figure.dpi": 150})

lbdir, ngdir = sys.argv[1], sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else "."

def history(d):
    h = np.loadtxt(os.path.join(d, "history_ep0000000.dat"))
    return h[:, 0], h[:, 2]   # epoch, training cost per point

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))

# ---- panel (a): the two cold-start objectives ----------------------
for d, lab, c in [(lbdir, "full-batch L-BFGS ($m=40$)", "C0"),
                  (ngdir, "dual natural gradient (minibatch 120)", "C3")]:
    ep, cost = history(d)
    # log abscissa, as in the other convergence figures: half the descent
    # happens in the first few per cent of the epoch count
    ax[0].loglog(ep, cost, c, label=lab)
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("objective per point")
ax[0].set_title("(a) cold start from random weights", fontsize=9)
ax[0].legend(frameon=False)

# ---- panel (b): parity of the five components (L-BFGS run) ---------
dat = np.loadtxt(os.path.join(lbdir, "output_set0002.dat"))
m = 5
prd, exa = dat[:, 2:2 + m], dat[:, 2 + m:2 + 2 * m]
labs = ["$\\phi$", "$\\rho$", "$u$", "$v$", "$p$"]
lo = min(prd.min(), exa.min()); hi = max(prd.max(), exa.max())
ax[1].plot([lo, hi], [lo, hi], "k-", lw=0.8)
r2s = []
for k, lab in enumerate(labs):
    r2 = 1.0 - np.mean((prd[:, k] - exa[:, k]) ** 2) / np.var(exa[:, k])
    r2s.append(r2)
    ax[1].plot(exa[:, k], prd[:, k], ".", ms=2.0,
               label=f"{lab}: ${r2:.5f}$")
ax[1].set_xlabel("manufactured solution"); ax[1].set_ylabel("PINN")
ax[1].set_title("(b) parity, L-BFGS run ($R^2$ per component)", fontsize=9)
ax[1].legend(frameon=False, markerscale=3, ncol=2, fontsize=7)

fig.tight_layout()
out = os.path.join(outdir, "fig_ehd.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for lab, r2 in zip(["phi", "rho", "u", "v", "p"], r2s):
    print(f"  {lab:3s} R2 = {r2:.5f}")
