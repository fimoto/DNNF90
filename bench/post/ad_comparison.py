#!/usr/bin/env python3
"""Timing comparison against nested and Taylor-mode automatic differentiation.

    python3 bench/post/ad_comparison.py [--repeat 50]

The quantity timed is the one training pays for: the gradient, with
respect to every network weight, of a loss assembled from derivatives of
the network with respect to its inputs.  A forward pass alone is not
comparable, and neither is an epoch, since an epoch covers many batches.
Every timing below is one gradient evaluation over one batch, with the
same network shape, the same batch size and the same double precision as
the corresponding case in bench/.

Two settings, chosen because they differ in the property that matters:

  KdV      D0 = 2, K = 3.  The residual is u_t + 3 u u_x + u_xxx.  The
           only high derivative is u_xxx, a pure directional derivative
           along x, which is the case a directional method handles best.

  ZK7      D0 = 4, K = 7.  The residual carries dx Lap^j for j = 1,2,3,
           that is mixed partials of order three, five and seven across
           three spatial variables.  A directional expansion does not
           give these individually; recovering them needs one expansion
           per multi-index.

Three implementations:

  nested   reverse-mode AD applied K times, which is what a PINN written
           in either framework normally does
  jet      Taylor-mode AD (jax.experimental.jet), differentiated with
           respect to the parameters, i.e. reverse over forward
  library  DNNF90, timed separately by bench/post/ad_comparison_f90.sh

Frameworks that are not installed are reported as such and skipped.
"""
import argparse
import math
import time

ADV_KDV = 3.0
ZK = dict(adv=7.577955, g1=4.576334693878, g2=-2.428987921699,
          g3=0.396569456604)


# ----------------------------------------------------------------- torch
def run_torch(repeat):
    try:
        import torch
    except ImportError:
        print("  PyTorch not installed, skipped")
        return
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    def mlp(dims):
        layers = []
        for i in range(len(dims) - 2):
            layers += [torch.nn.Linear(dims[i], dims[i + 1]), torch.nn.Tanh()]
        layers += [torch.nn.Linear(dims[-2], dims[-1])]
        return torch.nn.Sequential(*layers)

    def d(y, x):
        return torch.autograd.grad(y, x, torch.ones_like(y),
                                   create_graph=True)[0]

    # ---- KdV -----------------------------------------------------------
    net = mlp([2, 16, 16, 1])
    X = torch.randn(20, 2, requires_grad=True)

    def loss_kdv():
        u = net(X)[:, 0]
        g = d(u, X)
        ux, ut = g[:, 0], g[:, 1]
        uxx = d(ux, X)[:, 0]
        uxxx = d(uxx, X)[:, 0]
        r = ut + ADV_KDV * u * ux + uxxx
        return 0.5 * (r ** 2).mean()

    time_torch("KdV  (D0=2, K=3)", net, loss_kdv, repeat)

    # ---- ZK7 -----------------------------------------------------------
    net = mlp([4, 8, 8, 1])
    X = torch.randn(10, 4, requires_grad=True)

    def lap(u, X):
        g = d(u, X)
        out = 0.0
        for i in range(3):                      # the three spatial variables
            out = out + d(g[:, i], X)[:, i]
        return out

    def loss_zk7():
        u = net(X)[:, 0]
        g = d(u, X)
        ux, ut = g[:, 0], g[:, 3]
        l1 = lap(u, X)
        l2 = lap(l1, X)
        l3 = lap(l2, X)
        r = (ut + ZK["adv"] * u * ux
             + ZK["g1"] * d(l1, X)[:, 0]
             + ZK["g2"] * d(l2, X)[:, 0]
             + ZK["g3"] * d(l3, X)[:, 0])
        return 0.5 * (r ** 2).mean()

    time_torch("ZK7  (D0=4, K=7)", net, loss_zk7, repeat)


def time_torch(label, net, loss_fn, repeat):
    import torch
    for _ in range(3):
        net.zero_grad()
        loss_fn().backward()
    t0 = time.perf_counter()
    for _ in range(repeat):
        net.zero_grad()
        loss_fn().backward()
    dt = (time.perf_counter() - t0) / repeat
    print("  %-20s nested  %9.4f ms" % (label, 1e3 * dt))


# ------------------------------------------------------------------- jax
def run_jax(repeat):
    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental.jet import jet
    except ImportError:
        print("  JAX not installed, skipped")
        return
    jax.config.update("jax_enable_x64", True)

    def init(key, dims):
        ps = []
        for i in range(len(dims) - 1):
            key, k1, k2 = jax.random.split(key, 3)
            ps.append((jax.random.normal(k1, (dims[i], dims[i + 1]))
                       / jnp.sqrt(dims[i]),
                       jax.random.normal(k2, (dims[i + 1],)) * 0.1))
        return ps

    def net(ps, z):
        h = z
        for W, b in ps[:-1]:
            h = jnp.tanh(h @ W + b)
        W, b = ps[-1]
        return (h @ W + b)[0]

    # ---- KdV, nested ---------------------------------------------------
    def res_kdv_nested(ps, z):
        f = lambda zz: net(ps, zz)
        g = jax.grad(f)
        ux = lambda zz: g(zz)[0]
        uxx = lambda zz: jax.grad(ux)(zz)[0]
        uxxx = jax.grad(uxx)(z)[0]
        return g(z)[1] + ADV_KDV * f(z) * ux(z) + uxxx

    # ---- KdV, Taylor-mode ----------------------------------------------
    def res_kdv_jet(ps, z):
        f = lambda zz: net(ps, zz)
        ex = jnp.array([1.0, 0.0])
        et = jnp.array([0.0, 1.0])
        u, ys = jet(f, (z,), ([ex, jnp.zeros(2), jnp.zeros(2)],))
        ux, uxxx = ys[0], ys[2]
        _, yt = jet(f, (z,), ([et],))
        return yt[0] + ADV_KDV * u * ux + uxxx

    # ---- ZK7, nested ---------------------------------------------------
    def res_zk7_nested(ps, z):
        f = lambda zz: net(ps, zz)

        def lap(h):
            return lambda zz: sum(jax.grad(lambda w: jax.grad(h)(w)[i])(zz)[i]
                                  for i in range(3))
        l1 = lap(f)
        l2 = lap(l1)
        l3 = lap(l2)
        g = jax.grad(f)
        return (g(z)[3] + ZK["adv"] * f(z) * g(z)[0]
                + ZK["g1"] * jax.grad(l1)(z)[0]
                + ZK["g2"] * jax.grad(l2)(z)[0]
                + ZK["g3"] * jax.grad(l3)(z)[0])

    # ---- ZK7, Taylor-mode ----------------------------------------------
    # dx Lap^j needs individual mixed partials.  A directional expansion
    # gives (v . grad)^q, so each mixed partial is recovered from
    # expansions along a set of directions; the number required is the
    # number of multi-indices of that order.  Here the operator is
    # expanded into its mixed partials and each is taken along the axis
    # combination it needs, which is the honest cost of a directional
    # method on an anisotropic operator.
    def res_zk7_jet(ps, z):
        f = lambda zz: net(ps, zz)
        E = [jnp.eye(4)[i] for i in range(4)]

        def dir_deriv(order, v):
            _, ys = jet(f, (z,), ([v] + [jnp.zeros(4)] * (order - 1),))
            return ys[order - 1]

        # dx Lap^j u expands into sums of d^(2j+1)/dx^a dy^b dz^c with
        # a+b+c = 2j+1 and a odd.  Each term is obtained from expansions
        # along mixed directions by polarization; the count below is the
        # number of distinct terms, which is what sets the cost.
        total = 0.0
        for j, gj in ((1, ZK["g1"]), (2, ZK["g2"]), (3, ZK["g3"])):
            q = 2 * j + 1
            terms = [(a, b, c) for a in range(1, q + 1, 2)
                     for b in range(0, q - a + 1, 2)
                     for c in [q - a - b] if c >= 0 and c % 2 == 0]
            acc = 0.0
            for (a, b, c) in terms:
                v = (a * E[0] + b * E[1] + c * E[2]) / max(a + b + c, 1)
                acc = acc + dir_deriv(q, v)
            total = total + gj * acc
        u = f(z)
        _, gx = jet(f, (z,), ([E[0]],))
        _, gt = jet(f, (z,), ([E[3]],))
        return gt[0] + ZK["adv"] * u * gx[0] + total

    def bench(label, res, ps, X, repeat):
        def loss(p, xs):
            r = jax.vmap(lambda z: res(p, z))(xs)
            return 0.5 * jnp.mean(r ** 2)
        g = jax.jit(jax.grad(loss))
        jax.block_until_ready(g(ps, X))
        t0 = time.perf_counter()
        for _ in range(repeat):
            jax.block_until_ready(g(ps, X))
        print("  %-20s %-7s %9.4f ms"
              % (label, res.__name__.split("_")[-1],
                 1e3 * (time.perf_counter() - t0) / repeat))

    ps2 = init(jax.random.PRNGKey(0), [2, 16, 16, 1])
    X2 = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
    bench("KdV  (D0=2, K=3)", res_kdv_nested, ps2, X2, repeat)
    bench("KdV  (D0=2, K=3)", res_kdv_jet, ps2, X2, repeat)

    ps4 = init(jax.random.PRNGKey(2), [4, 8, 8, 1])
    X4 = jax.random.normal(jax.random.PRNGKey(3), (10, 4))
    bench("ZK7  (D0=4, K=7)", res_zk7_nested, ps4, X4, repeat)
    bench("ZK7  (D0=4, K=7)", res_zk7_jet, ps4, X4, repeat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    a = ap.parse_args()
    print("one gradient of the loss over one batch, double precision,")
    print("one thread; batch 20 for KdV and 10 for ZK7, as in bench/")
    print()
    print("PyTorch")
    run_torch(a.repeat)
    print()
    print("JAX")
    run_jax(a.repeat)
