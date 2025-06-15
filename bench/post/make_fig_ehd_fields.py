"""Field plots for the electrohydrodynamic section: manufactured
solution, PINN, and pointwise error for all five components.

    python3 make_fig_ehd_fields.py RUN_DIR [outdir]

RUN_DIR holds output_set0002.dat of a finished cold-start run
(Exact_solution 1, so the exact fields ride in the file).
Writes fig_ehd_fields.pdf.
"""
import sys, os
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8,
                     "figure.dpi": 150})

run = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else "."
d = np.loadtxt(os.path.join(run, "output_set0002.dat"))
x, y = d[:, 0], d[:, 1]
m = 5
prd, exa = d[:, 2:2+m], d[:, 2+m:2+2*m]

# The collocation points are drawn at random inside the square
# [-pi, pi]^2 (0.98 pi for the interior set), so the convex hull of the
# cloud stops short of the corners and tricontourf leaves them unfilled.
# The four true corners of the domain are therefore added: the exact
# panel takes the manufactured solution there, and the network panel
# takes the network itself evaluated there, read from the run's
# nn_weight.dat.  Both panels then show what they claim to show at
# every plotted point, and the error panel is a true error at the
# corners too.  Nothing else uses these four points.
def _exact(px, py):
    return np.array([np.sin(px) * np.sin(py),
                     np.cos(px) * np.cos(py) + 0.5,
                     -np.cos(px) * np.sin(py),
                     np.sin(px) * np.cos(py),
                     -0.25 * (np.cos(2 * px) + np.cos(2 * py))])

def _load_net(path):
    tok = open(path).read().split(); i = 1
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
            b[j] = row[0]; W[j] = row[1:]
        Ws.append(W); bs.append(b)
    return Ws, bs

def _net(Ws, bs, X):
    Z = np.asarray(X, float).T
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.tanh(W @ Z + b[:, None])
    return (Ws[-1] @ Z + bs[-1][:, None]).T

L = np.pi
cx = np.array([-L, L, -L, L]);  cy = np.array([-L, -L, L, L])
cex = np.array([_exact(a, b) for a, b in zip(cx, cy)])
Ws, bs = _load_net(os.path.join(run, "nn_weight.dat"))
cprd = _net(Ws, bs, np.column_stack([cx, cy]))
x = np.concatenate([x, cx]);  y = np.concatenate([y, cy])
exa = np.vstack([exa, cex]);  prd = np.vstack([prd, cprd])

tri = mtri.Triangulation(x, y)
labs = ["$\\phi$", "$\\rho$", "$u$", "$v$", "$p$"]

fig, ax = plt.subplots(m, 3, figsize=(6.8, 9.6), sharex=True, sharey=True)
for k in range(m):
    vmin = min(exa[:, k].min(), prd[:, k].min())
    vmax = max(exa[:, k].max(), prd[:, k].max())
    for j, F in enumerate([exa[:, k], prd[:, k]]):
        im = ax[k, j].tricontourf(tri, F, levels=41, cmap="RdBu_r",
                                  vmin=vmin, vmax=vmax)
        for c in im.collections if hasattr(im, "collections") else []:
            c.set_rasterized(True)
        im.set_rasterized(True)
    fig.colorbar(im, ax=ax[k, :2], shrink=0.85, pad=0.015)
    err = np.abs(prd[:, k] - exa[:, k])
    # The error is small and its scale differs by rows, so the tick
    # labels would carry a common factor.  Matplotlib puts that factor
    # above the colour bar, where the row of panels above it leaves no
    # room and it is clipped; the exponent is folded into the bar label
    # instead and the ticks are left as plain numbers.
    emax = float(err.max())
    expo = 0 if emax <= 0 else int(np.floor(np.log10(emax)))
    scale = 10.0 ** expo
    im2 = ax[k, 2].tricontourf(tri, err / scale, levels=41, cmap="viridis")
    im2.set_rasterized(True)
    cb = fig.colorbar(im2, ax=ax[k, 2], shrink=0.85, pad=0.03)
    cb.set_label(r"$\times 10^{%d}$" % expo, fontsize=8, labelpad=2)
    ax[k, 0].set_ylabel(labs[k] + "\n$y$")
for j, ttl in enumerate(["manufactured", "PINN", "$|$error$|$"]):
    ax[0, j].set_title(ttl, fontsize=9)
for j in range(3):
    ax[m-1, j].set_xlabel("$x$")
out = os.path.join(outdir, "fig_ehd_fields.pdf")
fig.savefig(out, bbox_inches="tight", dpi=300)
print("wrote", out)
