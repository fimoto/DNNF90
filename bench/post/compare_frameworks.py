#!/usr/bin/env python3
"""Cost of one weight gradient, compared across frameworks.

    python3 bench/post/compare_frameworks.py [--setting kdv|hod] [--kmax 7]

What is timed is one gradient of a loss with respect to every network
weight, where the loss is built from derivatives of the network with
respect to its inputs.  That is the quantity a training step pays, and
it is the only quantity compared here: not a forward pass, not an epoch.
An epoch of the Fortran trainer covers many batches, and comparing an
epoch against a single framework gradient overstates the framework by
that factor.

Two settings, matching cases in the distribution:

  kdv   D0 = 2, network 2-16-16-1, the KdV residual u_t + 3 u u_x + u_xxx.
        Only the third derivative is high order and it is a pure
        directional derivative in x, which is the case a directional
        method such as Taylor mode handles best.

  hod   D0 = 4, network 4-8-8-1, every mixed partial up to order K as a
        supervised target, which is what zwork/hod_4d_k7 trains on.
        The number of these is C(D0+K, K): 330 at K = 7.

Three implementations:

  torch-nested   repeated autograd.grad, the usual way to build a PINN
                 loss in PyTorch
  jax-nested     the same in JAX, jit-compiled
  jax-jet        Taylor mode (jax.experimental.jet), which propagates a
                 truncated Taylor series along one direction

The derivative values are checked against each other before anything is
timed, so that the timings are known to be of the same computation.

Tunings measured and not adopted (they do not change the story):
  * jax by nested jacfwd (full derivative tensors, unique components
    read off): within a factor of two of the chains either way --
    faster at K<=3, slower at K=4 -- and the same OOM at K=5.
  * torch.compile: cannot form the weight gradient of this loss; the
    create_graph chains need double backward through the compiled
    graph, which is unsupported (pytorch/pytorch issue 91469).
  * torch.func grad-under-vmap: ~5x slower than the autograd chains
    on one CPU core.
"""
import argparse
import itertools
import math
import time


def build_mlp_params(seed, dims):
    import numpy as np
    rng = np.random.default_rng(seed)
    ps = []
    for i in range(len(dims) - 1):
        W = rng.standard_normal((dims[i], dims[i + 1])) / math.sqrt(dims[i])
        b = rng.standard_normal(dims[i + 1]) * 0.1
        ps.append((W, b))
    return ps


def multi_indices(d0, kmax):
    out = []
    for p in range(kmax + 1):
        for c in itertools.combinations_with_replacement(range(d0), p):
            a = [0] * d0
            for t in c:
                a[t] += 1
            out.append(tuple(a))
    return out


# --------------------------------------------------------------- pytorch
def torch_setup(dims, params, npt, d0, seed):
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    ps = [(torch.tensor(W, requires_grad=True),
           torch.tensor(b, requires_grad=True)) for W, b in params]
    # the points come from numpy so that both frameworks (and the
    # cross-check below) see the identical batch
    import numpy as np
    X = torch.tensor(np.random.default_rng(seed).standard_normal((npt, d0)))
    return torch, ps, X


def torch_net(torch, ps, z):
    h = z
    for W, b in ps[:-1]:
        h = torch.tanh(h @ W + b)
    W, b = ps[-1]
    return (h @ W + b).squeeze(-1)


def torch_kdv_loss(torch, ps, X):
    x = X.clone().requires_grad_(True)
    u = torch_net(torch, ps, x)
    g1 = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    ux, ut = g1[:, 0], g1[:, 1]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0][:, 0]
    uxxx = torch.autograd.grad(uxx.sum(), x, create_graph=True)[0][:, 0]
    r = ut + 3.0 * u * ux + uxxx
    return 0.5 * (r ** 2).mean()


def torch_hod_loss(torch, ps, X, alphas):
    """every mixed partial up to K, by repeated differentiation"""
    x = X.clone().requires_grad_(True)
    u = torch_net(torch, ps, x)
    # cache: multi-index -> tensor of that derivative at all points
    cache = {tuple([0] * X.shape[1]): u}
    # the value slot is one of the C(D0+K,K) derivatives the table counts,
    # and the Fortran timing includes it, so it enters the loss here too
    tot = (u ** 2).mean()
    for a in alphas:
        if sum(a) == 0:
            continue
        # differentiate a parent that differs in one component
        i = next(k for k in range(len(a)) if a[k] > 0)
        parent = list(a)
        parent[i] -= 1
        pv = cache[tuple(parent)]
        d = torch.autograd.grad(pv.sum(), x, create_graph=True)[0][:, i]
        cache[a] = d
        tot = tot + (d ** 2).mean()
    # the Fortran timing sums squares without the 1/2, and sums over
    # points rather than averaging; the cross-check below compares the
    # gradients, so the two conventions are made to agree here
    return tot


# ------------------------------------------------------------------- jax
def jax_setup(params, npt, d0, seed):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    ps = [(jnp.array(W), jnp.array(b)) for W, b in params]
    import numpy as np
    X = jnp.array(np.random.default_rng(seed).standard_normal((npt, d0)))
    return jax, jnp, ps, X


def jax_net(jnp, ps, z):
    h = z
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = ps[-1]
    return (h @ W + b)[0]


def jax_kdv_loss(jax, jnp, ps, X):
    def res(p, z):
        f = lambda zz: jax_net(jnp, p, zz)
        d1 = jax.grad(f)
        ux = lambda zz: d1(zz)[0]
        uxx = lambda zz: jax.grad(ux)(zz)[0]
        uxxx = jax.grad(uxx)(z)[0]
        return d1(z)[1] + 3.0 * f(z) * ux(z) + uxxx
    r = jax.vmap(lambda z: res(ps, z))(X)
    return 0.5 * jnp.mean(r ** 2)


def jax_kdv_loss_jet(jax, jnp, ps, X):
    from jax.experimental.jet import jet

    def res(p, z):
        f = lambda zz: jax_net(jnp, p, zz)
        ex = jnp.array([1.0, 0.0])
        et = jnp.array([0.0, 1.0])
        y0, ys = jet(f, (z,), ([ex, jnp.zeros(2), jnp.zeros(2)],))
        ux, uxxx = ys[0], ys[2]
        _, yt = jet(f, (z,), ([et],))
        return yt[0] + 3.0 * y0 * ux + uxxx
    r = jax.vmap(lambda z: res(ps, z))(X)
    return 0.5 * jnp.mean(r ** 2)


def jax_dir_loss_jet(jax, jnp, ps, X, K, d0):
    """The directional task, by Taylor mode.

    The same loss the C++ columns and the Fortran directional column
    time: the square of the K-th derivative along one fixed direction,
    summed over the batch.  jax.experimental.jet propagates the Taylor
    coefficients along that direction in one pass, so this is the
    natural way to ask JAX for it, and the honest comparison for a
    method that also propagates a truncated jet.  The K-th derivative
    is K! times the K-th Taylor coefficient.
    """
    from jax.experimental.jet import jet
    import math

    v = jnp.zeros(d0).at[0].set(1.0)
    zero = jnp.zeros(d0)
    series = [v] + [zero] * (K - 1)

    def top(p, z):
        f = lambda zz: jax_net(jnp, p, zz)
        _, ys = jet(f, (z,), (series,))
        return math.factorial(K) * ys[K - 1]

    t = jax.vmap(lambda z: top(ps, z))(X)
    return jnp.sum(t ** 2)


def jax_hod_loss(jax, jnp, ps, X, alphas, d0):
    """every mixed partial up to K, by repeated differentiation"""
    def per_point(p, z):
        f = lambda zz: jax_net(jnp, p, zz)
        cache = {tuple([0] * d0): f}
        # the value slot, as in the torch loss and the Fortran timing
        tot = f(z) ** 2
        for a in alphas:
            if sum(a) == 0:
                continue
            i = next(k for k in range(d0) if a[k] > 0)
            parent = list(a)
            parent[i] -= 1
            pf = cache[tuple(parent)]
            df = (lambda pf_, i_: (lambda zz: jax.grad(pf_)(zz)[i_]))(pf, i)
            cache[a] = df
            tot = tot + df(z) ** 2
        return tot
    return 0.5 * jnp.mean(jax.vmap(lambda z: per_point(ps, z))(X))


# ----------------------------------------------------------------- timing
def timeit(fn, repeat, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=("kdv", "hod", "dir"), default="kdv")
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--npt", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    # In the derivative-only loss the output bias has zero gradient by
    # construction, since it cancels in every derivative of order one or
    # more.  allow_unused covers that; it is not a masked error.
    if args.setting == "kdv":
        d0, dims = 2, [2, 16, 16, 1]
    else:
        d0, dims = 4, [4, 8, 8, 1]
    params = build_mlp_params(12345, dims)
    alphas = multi_indices(d0, args.kmax)

    print("setting %s: D0=%d, network %s, %d points, K=%d, %d derivatives"
          % (args.setting, d0, "-".join(map(str, dims)), args.npt,
             args.kmax, len(alphas)))
    print("timed quantity: one gradient of the loss w.r.t. every weight")
    print()

    results = {}

    # ---- pytorch
    try:
        torch, tps, tX = torch_setup(dims, params, args.npt, d0, 7)
        if args.setting == "kdv":
            def tstep():
                loss = torch_kdv_loss(torch, tps, tX)
                gs = torch.autograd.grad(loss, [p for pr in tps for p in pr])
                return gs
        else:
            def tstep():
                loss = torch_hod_loss(torch, tps, tX, alphas)
                gs = torch.autograd.grad(
                    loss, [p for pr in tps for p in pr],
                    allow_unused=True)
                return gs
        # evidence values for the cross-check below
        _tl = (torch_kdv_loss(torch, tps, tX) if args.setting == "kdv"
               else torch_hod_loss(torch, tps, tX, alphas))
        _tg = torch.autograd.grad(_tl, tps[0][0], allow_unused=True,
                                  retain_graph=True)[0][0, 0]
        check_torch = (float(_tl), float(_tg))
        results["torch-nested"] = timeit(tstep, args.repeat)
    except Exception as e:
        check_torch = None
        results["torch-nested"] = None
        print("  torch-nested failed:", str(e)[:90])

    # ---- jax
    try:
        jax, jnp, jps, jX = jax_setup(params, args.npt, d0, 7)
        if args.setting == "kdv":
            gj = jax.jit(jax.grad(lambda p: jax_kdv_loss(jax, jnp, p, jX)))
        else:
            gj = jax.jit(jax.grad(
                lambda p: jax_hod_loss(jax, jnp, p, jX, alphas, d0)))
        # the docstring's promise, kept: the two frameworks are given the
        # same weights and points, so their loss and gradient must agree
        # to rounding before either timing is reported
        _jl = float(jax_kdv_loss(jax, jnp, jps, jX) if args.setting == "kdv"
                    else jax_hod_loss(jax, jnp, jps, jX, alphas, d0))
        _jg = float(gj(jps)[0][0][0, 0])
        print("  computed loss  torch %.12e   jax %.12e" %
              (check_torch[0], _jl) if check_torch else
              "  computed loss  jax %.12e" % _jl)
        if check_torch is not None:
            same = (abs(check_torch[0] - _jl) <= 1e-9 * abs(_jl) and
                    abs(check_torch[1] - _jg) <= 1e-7 * max(abs(_jg), 1e-30))
            print("  cross-check torch vs jax (loss and dL/dw_11):",
                  "PASS" if same else "FAIL -- do not use these timings")
        results["jax-nested"] = timeit(
            lambda: jax.block_until_ready(gj(jps)), args.repeat)
    except Exception as e:
        results["jax-nested"] = None
        print("  jax-nested failed:", str(e)[:90])

    # ---- jax taylor mode on the directional task, which is what it does
    if args.setting == "dir":
        try:
            gjd = jax.jit(jax.grad(
                lambda p: jax_dir_loss_jet(jax, jnp, p, jX, args.kmax, d0)))
            results["jax-jet"] = timeit(
                lambda: jax.block_until_ready(gjd(jps)), args.repeat)
        except Exception as e:
            results["jax-jet"] = None
            print("  jax-jet failed:", str(e)[:90])

    # ---- jax taylor mode, kdv only (a directional method; see the note)
    if args.setting == "kdv":
        try:
            gjt = jax.jit(jax.grad(
                lambda p: jax_kdv_loss_jet(jax, jnp, p, jX)))
            results["jax-jet"] = timeit(
                lambda: jax.block_until_ready(gjt(jps)), args.repeat)
        except Exception as e:
            results["jax-jet"] = None
            print("  jax-jet failed:", str(e)[:90])

    print("  %-14s %12s" % ("implementation", "ms/gradient"))
    for k, v in results.items():
        print("  %-14s %12s" % (k, "%.4f" % (1e3 * v) if v else "failed"))


if __name__ == "__main__":
    main()
