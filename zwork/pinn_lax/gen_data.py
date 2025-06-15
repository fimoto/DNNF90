#!/usr/bin/env python3
"""Regenerate data.dat and colloc.dat of the Lax case (fixed seed).
u = 2k^2 sech^2(k(x-16k^4 t)), k = 1/2; x in [-8,8], t in [0,2]."""
import numpy as np
rng = np.random.default_rng(7)
k = 0.5; c = 16*k**4
u = lambda x,t: 2*k**2/np.cosh(k*(x-c*t))**2
xi = np.linspace(-8,8,60); tb = np.linspace(0,2,20)
X = np.concatenate([xi, -8*np.ones(20), 8*np.ones(20)])
T = np.concatenate([np.zeros(60), tb, tb])
with open("data.dat","w") as f:
    for a,b in zip(X,T): f.write(f"{a:22.14e} {b:22.14e} {u(a,b):22.14e}\n")
xc = rng.uniform(-8,8,200); tc = rng.uniform(0,2,200)
with open("colloc.dat","w") as f:
    for a,b in zip(xc,tc): f.write(f"{a:22.14e} {b:22.14e} {u(a,b):22.14e}\n")
print("regenerated")
