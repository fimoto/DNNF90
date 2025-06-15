# This file (bench/post/torch_pinn.py) is part of DNNF90.
# (MIT License; see LICENSE at the repository root.)
#
# PyTorch baseline for the per-epoch cost comparison with DNNF90
# (Table "Comparison with PyTorch nested automatic differentiation" in
# the preprint).  Measured with torch 2.13.0 (CPU) on one thread.
#
# Usage (from bench/post, after the benchmark data exist):
#     pip install torch
#     python3 torch_pinn.py kdv 2000     # K=3, prints ms/epoch
#     python3 torch_pinn.py g7  200      # K=7, prints ms/epoch
# Replicates the DNNF90 PINN setup: same network (2-16-16-1 tanh, linear
# output), same fixed data/collocation sets, same batch size, same Adam
# settings, same loss weighting. High-order derivatives are obtained by
# nested torch.autograd.grad calls, which is the standard PyTorch approach.
# Single CPU thread for a like-for-like single-core comparison.
import sys, time
import numpy as np
import torch

torch.set_num_threads(1)
torch.manual_seed(20260728)
case = sys.argv[1]            # 'kdv' or 'g7'
nepoch = int(sys.argv[2]) if len(sys.argv) > 2 else 500

dat = np.loadtxt(f"../{case}/data.dat")     # x t u
col = np.loadtxt(f"../{case}/colloc.dat")   # x t u_exact
Xd = torch.tensor(dat[:, :2], dtype=torch.float64)
Yd = torch.tensor(dat[:, 2:3], dtype=torch.float64)
Xc = torch.tensor(col[:, :2], dtype=torch.float64)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 16), torch.nn.Tanh(),
    torch.nn.Linear(16, 1)).double()

def dx_n(u, x, n, idx):
    # n-th derivative w.r.t. input component idx by nested autograd
    d = u
    for _ in range(n):
        d = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0][:, idx:idx+1]
    return d

def residual(x):
    x = x.requires_grad_(True)
    u = net(x)
    ut = dx_n(u, x, 1, 1)
    ux = dx_n(u, x, 1, 0)
    if case == "kdv":
        u3 = dx_n(u, x, 3, 0)
        return ut + 3.0 * u * ux + u3
    u3 = dx_n(u, x, 3, 0)
    u5 = dx_n(u, x, 5, 0)
    u7 = dx_n(u, x, 7, 0)
    return ut + 7.577955 * u * ux + 6.2289 * u3 - 4.5 * u5 + u7

wr = 0.2                      # GD_ratio of the residual set
opt = torch.optim.Adam(net.parameters(), lr=3e-3, betas=(0.9, 0.999), eps=1e-8)
nb = 20
nd, nc = Xd.shape[0], Xc.shape[0]

t0 = time.perf_counter()
for ep in range(nepoch):
    id_ = torch.randint(0, nd, (nb,))
    ic_ = torch.randint(0, nc, (nb,))
    opt.zero_grad()
    ud = net(Xd[id_])
    loss = 0.5 * ((ud - Yd[id_]) ** 2).mean() \
         + wr * 0.5 * (residual(Xc[ic_]) ** 2).mean()
    loss.backward()
    opt.step()
t1 = time.perf_counter()
print(f"{case}: {nepoch} epochs {t1-t0:.3f} s -> {1e3*(t1-t0)/nepoch:.3f} ms/epoch  final_loss={loss.item():.3e}")
