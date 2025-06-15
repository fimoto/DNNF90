"""Load a DNNF90 nn_weight.dat and evaluate the tanh network; exact solutions."""
import numpy as np

def load_net(path):
    with open(path) as f:
        tok = f.read().split()
    i = 0
    _epoch = int(tok[i]); i += 1
    assert tok[i] == "func"; i += 2          # func 0 (tanh)
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
    """X: (N, D0) -> (N,) network output (tanh hidden, identity output)."""
    Z = np.asarray(X, float).T
    for W, b in zip(Ws[:-1], bs[:-1]):
        Z = np.tanh(W @ Z + b[:, None])
    Z = Ws[-1] @ Z + bs[-1][:, None]
    return Z[0]

# amplitude-normalised exact solitons of the benchmark grid
CASES = {
    "kdv":      dict(D0=2, k=0.5,  c=1.0,    p=2),
    "kawahara": dict(D0=2, k=0.3,  c=4.6656, p=4),
    "g7":       dict(D0=2, k=0.15, c=2.6244, p=6),
    "zk3":      dict(D0=4, k=0.5,  c=1.0,    p=2),
    "zk5":      dict(D0=4, k=0.3,  c=4.6656, p=4),
    "zk7":      dict(D0=4, k=0.15, c=2.6244, p=6),
}

def u_exact(name, X):
    prm = CASES[name]; X = np.asarray(X, float)
    if prm["D0"] == 2:
        xi = X[:, 0] - prm["c"] * X[:, 1]
    else:
        xi = X[:, 0] + 0.5 * X[:, 1] + X[:, 2] / 3.0 - prm["c"] * X[:, 3]
    return 1.0 / np.cosh(prm["k"] * xi) ** prm["p"]
