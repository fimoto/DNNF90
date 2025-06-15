"""Figure for the Lax subsection: profiles and the error field.

Run from zwork/pinn_lax after the run:
    python3 ../../bench/post/make_fig_lax.py [outdir]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                     "legend.fontsize": 8, "lines.linewidth": 1.3,
                     "figure.dpi": 150})
# fifth-order case: u = 2k^2 sech^2(k(x-16k^4 t)), k = 1/2
# seventh-order case (--lax7): u = (1/2) sech^2((x-t)/2)
LAX7 = "--lax7" in sys.argv
if LAX7: sys.argv.remove("--lax7")
K = 0.5
C = 1.0 if LAX7 else 16*K**4
A = 0.5 if LAX7 else 2*K**2
def u_exact(x, t): return A/np.cosh(K*(x - C*t))**2

def load_net(path):
    tok = open(path).read().split(); i = 0
    _ = int(tok[i]); i += 1
    assert tok[i] == "func"; i += 2
    assert tok[i] == "Activation_out"; i += 2
    L = int(tok[i]); i += 1
    dims = [int(tok[i+k]) for k in range(L)]; i += L
    Ws, bs = [], []
    for l in range(1, L):
        assert tok[i] == "#l="; i += 2
        W = np.empty((dims[l], dims[l-1])); b = np.empty(dims[l])
        for j in range(dims[l]):
            row = [float(tok[i+k]) for k in range(dims[l-1]+1)]
            i += dims[l-1]+1
            b[j] = row[0]; W[j,:] = row[1:]
        Ws.append(W); bs.append(b)
    return Ws, bs

def forward(Ws, bs, X):
    Z = np.asarray(X, float).T
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.tanh(W @ Z + b[:, None])
    return (Ws[-1] @ Z + bs[-1][:, None])[0]

outdir = sys.argv[1] if len(sys.argv) > 1 else "."
Ws, bs = load_net("nn_weight.dat")

fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5),
                       gridspec_kw={"width_ratios": [1, 1.25]})
# the window is taken from the case's own collocation cloud, so the
# figure follows the input rather than a hard-coded domain
_c = np.loadtxt("colloc.dat")
X0, X1 = _c[:, 0].min(), _c[:, 0].max()
T0, T1 = _c[:, 1].min(), _c[:, 1].max()
TS = [T0, 0.5 * (T0 + T1), T1]
x = np.linspace(X0, X1, 400)
for t0, c in zip(TS, ["C0", "C1", "C2"]):
    ax[0].plot(x, u_exact(x, t0), c, lw=2.2, alpha=0.35)
    X = np.column_stack([x, np.full_like(x, t0)])
    ax[0].plot(x, forward(Ws, bs, X), c + "--", label=f"$t={t0:g}$")
ax[0].set_xlabel("$x$"); ax[0].set_ylabel("$u$")
ax[0].set_title("(a) PINN (dashed) vs exact (solid)", fontsize=9)
ax[0].legend(frameon=False)

xg = np.linspace(X0, X1, 240)
tg = np.linspace(T0, T1, 120)
XX, TT = np.meshgrid(xg, tg, indexing="ij")
P = forward(Ws, bs, np.column_stack([XX.ravel(), TT.ravel()])).reshape(XX.shape)
E = np.abs(P - u_exact(XX, TT))
im = ax[1].pcolormesh(TT, XX, E, shading="auto", cmap="viridis",
                      rasterized=True)
fig.colorbar(im, ax=ax[1], label="$|u_{\\rm PINN}-u^*|$")
ax[1].set_xlabel("$t$"); ax[1].set_ylabel("$x$")
ax[1].set_title("(b) pointwise error", fontsize=9)

fig.tight_layout()
out = os.path.join(outdir, "fig_lax7.pdf" if LAX7 else "fig_lax.pdf")
fig.savefig(out, bbox_inches="tight", dpi=300)
print("wrote", out, "  max |err| = %.2e" % E.max())
