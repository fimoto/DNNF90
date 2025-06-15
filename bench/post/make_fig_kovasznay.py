"""Figure for the Kovasznay section: exact vs PINN profiles and parity.

Run from zwork/pinn_kovasznay after the two-stage run described in
zwork/FLOW_CASES.md (stage 2 leaves nn_weight.dat and
output_set0002.dat in the directory):

    python3 ../../bench/post/make_fig_kovasznay.py [outdir]

Writes fig_kovasznay.pdf to outdir (default: .).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.3,
                     "figure.dpi": 150})

LAM = -0.96374054  # Re = 40

def exact(x, y):
    x = np.broadcast_to(np.asarray(x, float), np.broadcast(x, y).shape)
    u = 1.0 - np.exp(LAM * x) * np.cos(2 * np.pi * y)
    v = LAM / (2 * np.pi) * np.exp(LAM * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1.0 - np.exp(2 * LAM * x))
    return u, v, p

def load_net(path):
    tok = open(path).read().split()
    i = 0
    _ = int(tok[i]); i += 1
    assert tok[i] == "func"; i += 2
    assert tok[i] == "Activation_out"; i += 2
    L = int(tok[i]); i += 1
    dims = [int(tok[i + k]) for k in range(L)]; i += L
    Ws, bs = [], []
    for l in range(1, L):
        assert tok[i] == "#l="; i += 2
        W = np.empty((dims[l], dims[l - 1])); b = np.empty(dims[l])
        for j in range(dims[l]):
            row = [float(tok[i + k]) for k in range(dims[l - 1] + 1)]
            i += dims[l - 1] + 1
            b[j] = row[0]; W[j, :] = row[1:]
        Ws.append(W); bs.append(b)
    return dims, Ws, bs

def forward(Ws, bs, X):
    """X: (N, D0) -> (m, N): all outputs of the tanh network."""
    Z = np.asarray(X, float).T
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.tanh(W @ Z + b[:, None])
    return Ws[-1] @ Z + bs[-1][:, None]

outdir = sys.argv[1] if len(sys.argv) > 1 else "."
dims, Ws, bs = load_net("nn_weight.dat")

# ---- panel (a): profiles along y at x = 0.5 ------------------------
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
y = np.linspace(-0.5, 1.5, 400)
x0 = 0.5
X = np.column_stack([np.full_like(y, x0), y])
pred = forward(Ws, bs, X)
ue, ve, pe = exact(x0, y)
for k, (ex, lab, c) in enumerate([(ue, "$u$", "C0"), (ve, "$v$", "C1"),
                                  (pe, "$p$", "C2")]):
    ax[0].plot(y, ex, c, lw=2.2, alpha=0.35)
    ax[0].plot(y, pred[k], c + "--", label=lab)
ax[0].set_xlabel("$y$"); ax[0].set_ylabel("$u,\\ v,\\ p$")
ax[0].set_title(f"(a) profiles at $x={x0}$: PINN (dashed) vs exact (solid)",
                fontsize=9)
ax[0].legend(frameon=False, ncol=3)

# ---- panel (b): parity at the collocation points -------------------
dat = np.loadtxt("output_set0002.dat")
m = 3
xy, prd, exa = dat[:, :2], dat[:, 2:2 + m], dat[:, 2 + m:2 + 2 * m]
lo = min(prd.min(), exa.min()); hi = max(prd.max(), exa.max())
ax[1].plot([lo, hi], [lo, hi], "k-", lw=0.8)
for k, lab in enumerate(["$u$", "$v$", "$p$"]):
    r2 = 1.0 - np.mean((prd[:, k] - exa[:, k]) ** 2) / np.var(exa[:, k])
    ax[1].plot(exa[:, k], prd[:, k], ".", ms=2.5,
               label=f"{lab}: $R^2={r2:.5f}$")
ax[1].set_xlabel("exact"); ax[1].set_ylabel("PINN")
ax[1].set_title("(b) parity at the collocation points", fontsize=9)
ax[1].legend(frameon=False, markerscale=3)

fig.tight_layout()
out = os.path.join(outdir, "fig_kovasznay.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for k, lab in enumerate(["u", "v", "p"]):
    r2 = 1.0 - np.mean((prd[:, k] - exa[:, k]) ** 2) / np.var(exa[:, k])
    print(f"  {lab}  R2 = {r2:.5f}")
