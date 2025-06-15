#!/usr/bin/env python3
# This file (bench/compare/grad_cost.py) is part of DNNF90.
# (MIT License; see LICENSE at the repository root.)
#
# What one training step costs, measured the same way in every framework.
#
# The quantity timed is the gradient of a loss with respect to every
# network weight, where the loss is built from derivatives of the network
# with respect to its inputs.  That is what a training step pays; a
# forward pass alone is not comparable, and neither is an "epoch", whose
# meaning differs between codes.  Every measurement here is one gradient
# evaluation over one batch of NPT points.
#
# Two settings:
#
#   kdv   D0 = 2, K = 3, network 2-16-16-1.  The residual needs u_t, u_x
#         and u_xxx.  Only u_xxx is high order and it is a pure
#         directional derivative, which is the case a directional method
#         such as Taylor-mode handles best.
#
#   hod7  D0 = 4, K = 7, network 4-8-8-1.  The loss is a fit to every
#         mixed partial up to order seven, 330 of them.  Nested
#         differentiation produces the full symmetric tensor at each
#         order, 4^q entries at order q against the C(q+3,3) distinct
#         ones, so the redundancy grows with the order.
#
# Usage:
#     python3 grad_cost.py torch kdv
#     python3 grad_cost.py jax   hod7
#     python3 grad_cost.py jax-jet kdv
#
# The DNNF90 side is timed by bench/compare/grad_cost.f90, which calls
# the same kernels the trainer uses.
import sys
import time

NPT = 20          # batch size, the same in every framework and in DNNF90
REPEAT = 50
SEED = 20260101

SETTINGS = {
    "kdv":  dict(d0=2, k=3, dims=[2, 16, 16, 1]),
    "hod7": dict(d0=4, k=7, dims=[4, 8, 8, 1]),
}


def multi_indices(d0, k):
    """all alpha with |alpha| <= k, in the order the library uses"""
    import itertools
    out = []
    for p in range(k + 1):
        if p == 0:
            out.append((0,) * d0)
            continue
        for combo in itertools.combinations_with_replacement(range(d0), p):
            a = [0] * d0
            for c in combo:
                a[c] += 1
            out.append(tuple(a))
    return out


# ------------------------------------------------------------------ torch
def run_torch(name):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    cfg = SETTINGS[name]
    d0, k, dims = cfg["d0"], cfg["k"], cfg["dims"]

    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1], dtype=torch.float64))
    net = torch.nn.ModuleList(layers)

    def f(z):
        h = z
        for L in list(net)[:-1]:
            h = torch.tanh(L(h))
        return list(net)[-1](h).squeeze(-1)

    X = torch.randn(NPT, d0, dtype=torch.float64, requires_grad=True)

    if name == "kdv":
        def loss():
            u = f(X)
            g = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
            ux, ut = g[:, 0], g[:, 1]
            uxx = torch.autograd.grad(ux.sum(), X, create_graph=True)[0][:, 0]
            uxxx = torch.autograd.grad(uxx.sum(), X, create_graph=True)[0][:, 0]
            r = ut + 3.0 * u * ux + uxxx
            return 0.5 * (r ** 2).mean()
    else:
        alphas = multi_indices(d0, k)
        target = torch.zeros(NPT, len(alphas), dtype=torch.float64)

        def loss():
            # build every mixed partial by repeated differentiation
            vals = {(0,) * d0: f(X)}
            for order in range(1, k + 1):
                new = {}
                for a, v in vals.items():
                    if sum(a) != order - 1:
                        continue
                    g = torch.autograd.grad(v.sum(), X, create_graph=True)[0]
                    for i in range(d0):
                        b = list(a)
                        b[i] += 1
                        b = tuple(b)
                        if b not in new:
                            new[b] = g[:, i]
                vals.update(new)
            stack = torch.stack([vals[a] for a in alphas], dim=1)
            return 0.5 * ((stack - target) ** 2).mean()

    params = [p for L in net for p in L.parameters()]

    def step():
        L = loss()
        return torch.autograd.grad(L, params)

    step()
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        step()
    return (time.perf_counter() - t0) / REPEAT


# -------------------------------------------------------------------- jax
def run_jax(name, mode):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    cfg = SETTINGS[name]
    d0, k, dims = cfg["d0"], cfg["k"], cfg["dims"]

    key = jax.random.PRNGKey(SEED)
    ps = []
    for i in range(len(dims) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        ps.append((jax.random.normal(k1, (dims[i], dims[i + 1]))
                   / jnp.sqrt(dims[i]),
                   jax.random.normal(k2, (dims[i + 1],)) * 0.1))

    def net(ps, z):
        h = z
        for W, b in ps[:-1]:
            h = jnp.tanh(h @ W + b)
        W, b = ps[-1]
        return (h @ W + b)[0]

    X = jax.random.normal(jax.random.PRNGKey(SEED + 1), (NPT, d0))

    if name == "kdv":
        if mode == "jet":
            from jax.experimental.jet import jet

            def res(ps, z):
                f = lambda zz: net(ps, zz)
                ex = jnp.eye(d0)[0]
                et = jnp.eye(d0)[1]
                y0, ys = jet(f, (z,), ([ex, jnp.zeros(d0), jnp.zeros(d0)],))
                _, yt = jet(f, (z,), ([et],))
                return yt[0] + 3.0 * y0 * ys[0] + ys[2]
        else:
            def res(ps, z):
                f = lambda zz: net(ps, zz)
                d1 = jax.grad(f)
                ux = lambda zz: d1(zz)[0]
                uxx = lambda zz: jax.grad(ux)(zz)[0]
                uxxx = lambda zz: jax.grad(uxx)(zz)[0]
                return d1(z)[1] + 3.0 * f(z) * ux(z) + uxxx(z)

        def loss(ps, X):
            r = jax.vmap(lambda z: res(ps, z))(X)
            return 0.5 * jnp.mean(r ** 2)
    else:
        if mode == "jet":
            raise SystemExit(
                "hod7 asks for every mixed partial; a directional expansion\n"
                "gives one direction at a time, so jet needs one call per\n"
                "direction plus a solve.  Not implemented: see the note in\n"
                "the paper.")

        alphas = multi_indices(d0, k)

        def all_partials(ps, z):
            f = lambda zz: net(ps, zz)
            # nested jacfwd builds the full symmetric tensor at each order
            fns = [f]
            for _ in range(k):
                fns.append(jax.jacfwd(fns[-1]))
            out = []
            for a in alphas:
                q = sum(a)
                t = fns[q](z)
                idx = []
                for i, c in enumerate(a):
                    idx += [i] * c
                for i in idx:
                    t = t[i]
                out.append(t)
            return jnp.array(out)

        def loss(ps, X):
            v = jax.vmap(lambda z: all_partials(ps, z))(X)
            return 0.5 * jnp.mean(v ** 2)

    g = jax.jit(jax.grad(loss))
    jax.block_until_ready(g(ps, X))
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        jax.block_until_ready(g(ps, X))
    return (time.perf_counter() - t0) / REPEAT


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    which, case = sys.argv[1], sys.argv[2]
    if case not in SETTINGS:
        sys.exit("case must be one of: " + ", ".join(SETTINGS))
    if which == "torch":
        dt = run_torch(case)
    elif which == "jax":
        dt = run_jax(case, "nested")
    elif which == "jax-jet":
        dt = run_jax(case, "jet")
    else:
        sys.exit("framework must be torch, jax or jax-jet")
    n = len(multi_indices(SETTINGS[case]["d0"], SETTINGS[case]["k"]))
    print("%-8s %-6s  %3d derivatives  %8.4f ms per gradient over %d points"
          % (which, case, n, 1e3 * dt, NPT))
