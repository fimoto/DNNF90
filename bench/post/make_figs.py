"""Generate all preprint figures and the benchmark table from bench/ results.

Run from the bench/ directory:  python3 post/make_figs.py
Outputs go to bench/figs/ :
  fig_solution.pdf     NN vs exact soliton profiles (KdV and 7th-order ZK)
  fig_convergence.pdf  residual cost and u-RMSE vs epoch, all six cases
  fig_optimizers.pdf   optimizer comparison on the KdV case
  fig_scaling.pdf      wall time per epoch, dense vs active-set closure (ZK7)
  bench_table.tex      summary table (LaTeX, booktabs)
  summary.txt          plain-text record of every number used in the paper
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nn_eval import load_net, forward, u_exact, CASES

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "lines.linewidth": 1.3, "figure.dpi": 150})
os.makedirs("figs", exist_ok=True)

GRID = ["kdv", "kawahara", "g7", "zk3", "zk5", "zk7"]
LABEL = {"kdv": "KdV (2 var, 3rd)", "kawahara": "Kawahara (2 var, 5th)",
         "g7": "7th-order (2 var, 7th)", "zk3": "ZK (4 var, 3rd)",
         "zk5": "ext. ZK (4 var, 5th)", "zk7": "ext. ZK (4 var, 7th)"}
ORDER = {"kdv": 3, "kawahara": 5, "g7": 7, "zk3": 3, "zk5": 5, "zk7": 7}

def curves(case):
    """epochs, residual train cost, u-RMSE on the collocation set (set 2)."""
    hist = np.loadtxt(f"{case}/history_ep0000000.dat")
    # 0 epoch, 2 cost_train; after column 22 come blocks of the same 21
    # metrics per set, and rmse_train_f is the 10th metric, so the
    # collocation set (set 2) has it at 2 + 21 + 21 + 9 = 53
    return hist[:, 0], hist[:, 2], hist[:, 53]

def log_info(case):
    txt = open(f"{case}/a.log").read()
    p = re.findall(r"\[(?:P|dW-R)\].*= +([0-9.E+-]+)", txt)
    na = re.findall(r"NUM_alpha \(multi-indices carried\) = (\d+)", txt)
    t = re.findall(r"([0-9.]+)\[s\]", txt)
    return (float(p[-1]) if p else np.nan,
            int(na[-1]) if na else 0,
            float(t[-1]) if t else np.nan)

summary = []

# ------------------------------------------------------------------ fig 1
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
dims, Ws, bs = load_net("kdv/nn_weight.dat")
x = np.linspace(-8, 8, 400)
for t, cstyle in [(0.0, "C0"), (1.0, "C1"), (2.0, "C2")]:
    X = np.column_stack([x, np.full_like(x, t)])
    ax[0].plot(x, u_exact("kdv", X), cstyle, lw=2.2, alpha=0.35)
    ax[0].plot(x, forward(Ws, bs, X), cstyle + "--", label=f"$t={t:g}$")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u$")
ax[0].set_title("(a) KdV: PINN (dashed) vs exact (solid)", fontsize=9)
ax[0].legend(frameon=False)

dims, Ws, bs = load_net("zk7/nn_weight.dat")
s = np.linspace(-3, 3, 300); t7 = 0.5
X = np.column_stack([s, 0.5 * s * 0 + 0.0 * s, 0 * s, np.full_like(s, t7)])
# cut along x at y=z=0
ax[1].plot(s, u_exact("zk7", X), "k", lw=2.2, alpha=0.35, label="exact")
ax[1].plot(s, forward(Ws, bs, X), "C3--", label="PINN")
X2 = np.column_stack([0 * s, s, 0 * s, np.full_like(s, t7)])
ax[1].plot(s, u_exact("zk7", X2), "k", lw=2.2, alpha=0.35)
ax[1].plot(s, forward(Ws, bs, X2), "C0--", label="PINN ($y$ cut)")
ax[1].set_xlabel("$x$ (red) / $y$ (blue),  $y{=}z{=}0$ resp. $x{=}z{=}0$,  $t=0.5$")
ax[1].set_ylabel("$u$")
ax[1].set_title("(b) 7th-order ZK (4 variables)", fontsize=9)
ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("figs/fig_solution.pdf"); plt.close(fig)

# ------------------------------------------------------------------ fig 2
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
for c in GRID:
    ep, cost, rms = curves(c)
    ls = "-" if CASES[c]["D0"] == 2 else "--"
    # Log abscissa: most of the descent happens in the first few per cent
    # of the epoch count, which a linear axis crushes against the origin.
    ax[0].loglog(ep, cost, ls, label=LABEL[c])
    ax[1].loglog(ep, rms, ls)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training cost $C_0$")
ax[0].set_title("(a) total cost", fontsize=9)
ax[0].legend(frameon=False, fontsize=6.5)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel(r"$u$-RMSE (collocation)")
ax[1].set_title("(b) solution error vs exact soliton", fontsize=9)
fig.tight_layout(); fig.savefig("figs/fig_convergence.pdf"); plt.close(fig)

# ------------------------------------------------------------------ fig 3
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for c, lab in [("opt_simple", "SGD"), ("kdv", "Adam"), ("opt_ngd", "natural gradient"),
               ("opt_kalman", "Kalman filter")]:
    ep, cost, _ = curves(c)
    ax.loglog(ep, cost, label=lab)
# The runs differ by nearly three decades in length -- 100 epochs for the
# filter against 60,000 for Adam -- so a linear abscissa collapses the
# short curves against the axis and shows only the long ones.
ax.set_xlabel("epoch"); ax.set_ylabel("training cost $C_0$")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figs/fig_optimizers.pdf"); plt.close(fig)

# ------------------------------------------------------------------ fig 4
_, na_c, t_c = log_info("scal_closure")
_, na_d, t_d = log_info("scal_dense")
ep_scal = 500.0
fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.bar(["closure\n$|\\mathcal{A}|=%d$" % na_c,
               "dense\n$|\\mathcal{A}|=%d$" % na_d],
              [1e3 * t_c / ep_scal, 1e3 * t_d / ep_scal], color=["C0", "C3"], width=0.55)
ax.set_ylabel("wall time per epoch [ms]")
ax.bar_label(bars, fmt="%.2f")
ax.text(0.5, 0.75, r"$\times$%.1f" % (t_d / t_c), transform=ax.transAxes,
        ha="center", fontsize=12)
fig.tight_layout(); fig.savefig("figs/fig_scaling.pdf"); plt.close(fig)

# ------------------------------------------------------------------ table
rows = []
for c in GRID:
    ep, cost, rms = curves(c)
    p, na, t = log_info(c)
    # Report the best epoch, not the last one.  The weights that ship,
    # that the solution error is measured from and that early stopping
    # would select are the best-epoch weights, so the cost has to be the
    # cost of the same state; training is not monotone, and the last
    # validation event is an arbitrary point of the history.
    ib = int(np.argmin(cost))
    rows.append((LABEL[c], CASES[c]["D0"], ORDER[c], na, p, cost[ib], rms[ib], t))
    summary.append(f"{c}: |A|={na} [dW-R]={p:.2e} cost={cost[ib]:.2e} "
                   f"uRMSE={rms[ib]:.4f} time={t:.1f}s best_epoch={int(ep[ib])}")

def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

with open("figs/bench_table.tex", "w") as f:
    f.write("\\begin{tabular}{lccccccc}\n\\toprule\n"
            "equation & $D_0$ & order & $|\\mathcal{A}|$ & grad.\\ check &"
            " final cost & $u$-RMSE & time [s] \\\\\n\\midrule\n")
    for lab, d0, o, na, p, cst, rm, t in rows:
        f.write(f"{lab} & {d0} & {o} & {na} & {sci(p)} & {sci(cst)}"
                f" & {rm:.4f} & {t:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# literature-comparison cases (E & Yu 2018): relative L2 from the exact column
for c in ["eyu10d", "slit"]:
    if os.path.exists(f"{c}/history_ep0000000.dat"):
        rms = float(np.sqrt(np.mean(np.loadtxt(f"{c}/colloc.dat")[:, -1] ** 2)))
        ep, cost, rmsq = curves(c)
        p, na, t = log_info(c)
        summary.append(f"{c}: |A|={na} [dW-R]={p:.2e} relL2={rmsq[-1]/rms:.4f} "
                       f"(uRMSE={rmsq[-1]:.4f}, u_rms={rms:.3f}) time={t:.1f}s")

_, _, t_ngd = log_info("opt_ngd")
summary.append(f"scaling: closure {1e3*t_c/ep_scal:.2f} ms/ep vs dense "
               f"{1e3*t_d/ep_scal:.2f} ms/ep -> x{t_d/t_c:.1f}")
for c, n in [("opt_simple", "SGD"), ("kdv", "Adam"), ("opt_ngd", "NGD"),
             ("opt_kalman", "Kalman")]:
    ep, cost, _ = curves(c)
    ib = int(np.argmin(cost))
    summary.append(f"opt {n}: best cost {cost[ib]:.2e} at epoch {int(ep[ib])} "
                   f"of {int(ep[-1])}")
open("figs/summary.txt", "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
print("figures written to figs/")
