#!/usr/bin/env python3
"""Data for the five-component electrohydrodynamic benchmark.

The system couples an electrostatic field to an incompressible flow the
way a discharge-driven wind does: the charge density sets the potential
through Poisson's equation, the potential and the flow together move the
charge, and the charge acted on by the field is a body force on the
fluid.  That two-way coupling is the physics, and it is what a residual
of one field component alone cannot express.

    lap(phi) + rho/eps                              = S_phi
    u.grad(rho) - mu div(rho grad phi) - D lap(rho)  = S_rho
    u.grad(u) + p_x - nu lap(u) + c rho phi_x        = S_u
    u.grad(v) + p_y - nu lap(v) + c rho phi_y        = S_v
    u_x + v_y                                        = 0

The fields are chosen first and the sources follow, so the answer is
known exactly and the fit can be measured rather than guessed at.

    phi = sin x sin y
    rho = cos x cos y + 1/2
    u   = -cos x sin y      v = sin x cos y
    p   = -(cos 2x + cos 2y)/4

The charge density is deliberately NOT proportional to the potential.
Taking rho = -eps lap(phi) would make Poisson homogeneous and look
tidier, but it also makes two of the five components the same degree of
freedom: the residuals then constrain nearly the same direction, and
training diverges from a starting point that already fits the solution
to one percent.  Independence of the fields matters more than
homogeneity of one residual.
"""
import math
import random

EPS, MU, DIF, NU, CF = 1.0, 0.5, 0.1, 0.1, 0.2


def fields(x, y):
    phi = math.sin(x)*math.sin(y)
    rho = math.cos(x)*math.cos(y) + 0.5
    u = -math.cos(x)*math.sin(y)
    v = math.sin(x)*math.cos(y)
    p = -(math.cos(2*x) + math.cos(2*y))/4.0
    return phi, rho, u, v, p


def sources(x, y):
    s, c = math.sin(x), math.cos(x)
    S, C = math.sin(y), math.cos(y)
    phi_x, phi_y = c*S, s*C
    phi_xx, phi_yy = -s*S, -s*S
    rho = c*C + 0.5
    rho_x, rho_y = -s*C, -c*S
    rho_xx, rho_yy = -c*C, -c*C
    u, v = -c*S, s*C
    u_x, u_y = s*S, -c*C
    v_x, v_y = c*C, -s*S
    u_xx, u_yy = c*S, c*S
    v_xx, v_yy = -s*C, -s*C
    p_x, p_y = math.sin(2*x)/2.0, math.sin(2*y)/2.0

    s_phi = (phi_xx + phi_yy) + rho/EPS
    drgp = rho_x*phi_x + rho*phi_xx + rho_y*phi_y + rho*phi_yy
    s_rho = u*rho_x + v*rho_y - MU*drgp - DIF*(rho_xx + rho_yy)
    s_u = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy) + CF*rho*phi_x
    s_v = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy) + CF*rho*phi_y
    return s_phi, s_rho, s_u, s_v, 0.0


def main():
    random.seed(20260814)
    L = math.pi
    d = "zwork/pinn_ehd/"
    sup = []
    n_side = 40
    for k in range(n_side):
        t = -L + 2*L*(k + 0.5)/n_side
        sup += [(t, -L), (t, L), (-L, t), (L, t)]
    for _ in range(160):
        sup.append((random.uniform(-L, L), random.uniform(-L, L)))
    with open(d + "data.dat", "w") as f:
        for x, y in sup:
            f.write(" ".join("%22.14e" % q for q in (x, y) + fields(x, y)) + "\n")
    ncol = 1200
    with open(d + "colloc.dat", "w") as f:
        for _ in range(ncol):
            x = random.uniform(-0.98*L, 0.98*L)
            y = random.uniform(-0.98*L, 0.98*L)
            f.write(" ".join("%22.14e" % q
                             for q in (x, y) + sources(x, y) + fields(x, y)) + "\n")
    print("  %d supervised (%d boundary + %d interior), %d collocation"
          % (len(sup), 4*n_side, len(sup)-4*n_side, ncol))
    names = ["phi", "rho", "u", "v", "p"]
    acc = [0.0]*5
    for _ in range(4000):
        x = random.uniform(-L, L); y = random.uniform(-L, L)
        for i, q in enumerate(fields(x, y)):
            acc[i] += q*q
    print("  component scales, for Sys_wcomp:")
    for i, nm in enumerate(names):
        rms = (acc[i]/4000)**0.5
        print("    %-4s RMS %.4f   weight %.4f" % (nm, rms, 1.0/rms**2))


if __name__ == "__main__":
    main()
