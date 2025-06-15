#!/usr/bin/env python3
"""Data for the seventh-order Lax equation (Chen et al., Sci. Rep. 14
(2024) 23874, Eq. (20)), whose expanded form is their Eq. (53):

  u_t + 140 u^3 u_x + 70 u_x^3 + 280 u u_x u_xx + 70 u^2 u_xxx
      + 70 u_xx u_xxx + 42 u_x u_xxxx + 14 u u_xxxxx + u_xxxxxxx = 0

Exact solitary wave (their Eq. (39)): u = (1/2) sech^2((x - t)/2),
verified against the equation symbolically."""
import numpy as np
rng = np.random.default_rng(11)
u = lambda x, t: 0.5 / np.cosh(0.5 * (x - t)) ** 2
X0, X1, T0, T1 = -6.0, 8.0, 0.0, 2.0
xi = np.linspace(X0, X1, 60)
tb = np.linspace(T0, T1, 20)
X = np.concatenate([xi, np.full(20, X0), np.full(20, X1)])
T = np.concatenate([np.full(60, T0), tb, tb])
with open("data.dat", "w") as f:
    for a, b in zip(X, T):
        f.write(f"{a:22.14e} {b:22.14e} {u(a,b):22.14e}\n")
xc = rng.uniform(X0, X1, 200)
tc = rng.uniform(T0, T1, 200)
with open("colloc.dat", "w") as f:
    for a, b in zip(xc, tc):
        f.write(f"{a:22.14e} {b:22.14e} {u(a,b):22.14e}\n")
print("wrote data.dat (100 records) and colloc.dat (200 records)")
