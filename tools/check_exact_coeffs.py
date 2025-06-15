#!/usr/bin/env python3
"""Do the benchmark coefficients solve their equations exactly?

Section 5.1 states that the coefficients of the amplitude-normalised
hierarchy satisfy the equation to 30-40 digits, and Sections 5.5 and 5.6
give further closed-form solutions.  This script checks each of them
symbolically, so the claim rests on something runnable rather than on a
number quoted from a notebook.

    python3 tools/check_exact_coeffs.py        # needs sympy

Checked:
  * KdV, Kawahara and the seventh-order member of Eq. (13) with their
    sech^{2m} solitons, at the coefficients the bench inputs use
  * the plane-wave rule b = b1/L, d = d1/L^2, g = g1/L^3 that lifts a
    1+1 soliton to the 3+1 line soliton
  * the Kovasznay flow, Eq. (16), against the Navier-Stokes equations
  * the exact solitary wave of Lax's seventh-order equation, Eq. (15),
    and its expansion into nine terms
  * u = cosh(mu(x - mu^6 t)) against Eq. (19)
"""
import sympy as sp

xi, X, Y, Z, T = sp.symbols("xi X Y Z T")
k, A, nu, b, d, g, c = sp.symbols("k A nu b d g c", positive=True)
ok = True


def report(name, expr):
    global ok
    r = sp.simplify(expr)
    good = r == 0
    ok = ok and good
    print("  %-46s %s" % (name, "exact" if good else "NOT ZERO: %s" % r))


def hierarchy(power, coeffs, name):
    """-c u' + nu u u' + b u''' + d u^(5) + g u^(7) with u = A sech^power"""
    u = A * sp.sech(k * xi) ** power
    D = lambda q: sp.diff(u, xi, q)
    e = -c * D(1) + nu * u * D(1) + b * D(3) + d * D(5) + g * D(7)
    report(name, e.subs(coeffs).rewrite(sp.exp))


print("Eq. (13), the amplitude-normalised hierarchy:")
hierarchy(2, {nu: 12 * k**2 * b / A, c: 4 * k**2 * b, d: 0, g: 0},
          "KdV, u = A sech^2")
hierarchy(4, {nu: 1680 * k**4 / A, b: 52 * k**2, d: -1, g: 0,
              c: 576 * k**4}, "Kawahara, u = A sech^4")
hierarchy(6, {nu: 665280 * k**6 / A, b: 12304 * k**4, d: -200 * k**2,
              g: 1, c: 230400 * k**6}, "seventh order, u = A sech^6")

print("\nthe plane-wave lift to 3+1 dimensions:")
l, m, n = sp.symbols("l m n", positive=True)
L = l**2 + m**2 + n**2
U = sp.Function("U")
b1, d1, g1 = sp.symbols("b1 d1 g1")
arg = l * X + m * Y + n * Z - l * c * T
w = U(arg)
lap = lambda f: sp.diff(f, X, 2) + sp.diff(f, Y, 2) + sp.diff(f, Z, 2)
R31 = (sp.diff(w, T) + nu * w * sp.diff(w, X)
       + (b1 / L) * sp.diff(lap(w), X)
       + (d1 / L**2) * sp.diff(lap(lap(w)), X)
       + (g1 / L**3) * sp.diff(lap(lap(lap(w))), X))
s = sp.symbols("s")
R11 = l * (-c * U(s).diff(s) + nu * U(s) * U(s).diff(s)
           + b1 * U(s).diff(s, 3) + d1 * U(s).diff(s, 5)
           + g1 * U(s).diff(s, 7))
report("R(3+1) - l R(1+1)", R31 - R11.subs(s, arg).doit())

print("\nEq. (16), the Kovasznay flow at Re = 40:")
nu0 = sp.Rational(1, 40)
lam = 1 / (2 * nu0) - sp.sqrt(1 / (4 * nu0**2) + 4 * sp.pi**2)
print("  lambda = %s" % sp.N(lam, 12))
u = 1 - sp.exp(lam * X) * sp.cos(2 * sp.pi * Y)
v = lam / (2 * sp.pi) * sp.exp(lam * X) * sp.sin(2 * sp.pi * Y)
p = sp.Rational(1, 2) * (1 - sp.exp(2 * lam * X))
report("x momentum", u * u.diff(X) + v * u.diff(Y) + p.diff(X)
       - nu0 * (u.diff(X, 2) + u.diff(Y, 2)))
report("y momentum", u * v.diff(X) + v * v.diff(Y) + p.diff(Y)
       - nu0 * (v.diff(X, 2) + v.diff(Y, 2)))
report("continuity", u.diff(X) + v.diff(Y))

print("\nEq. (15), Lax's seventh-order equation:")
uL = sp.Rational(1, 2) * sp.sech((X - T) / 2) ** 2
dx = lambda f, q: sp.diff(f, X, q)
inner = (35 * uL**4 + 70 * (uL**2 * dx(uL, 2) + uL * dx(uL, 1) ** 2)
         + 7 * (2 * uL * dx(uL, 4) + 3 * dx(uL, 2) ** 2
                + 4 * dx(uL, 1) * dx(uL, 3)) + dx(uL, 6))
report("u = sech^2((x-t)/2)/2", (uL.diff(T) + sp.diff(inner, X)).rewrite(sp.exp))
u0 = sp.Function("u")(X)
inner0 = (35 * u0**4 + 70 * (u0**2 * u0.diff(X, 2) + u0 * u0.diff(X) ** 2)
          + 7 * (2 * u0 * u0.diff(X, 4) + 3 * u0.diff(X, 2) ** 2
                 + 4 * u0.diff(X) * u0.diff(X, 3)) + u0.diff(X, 6))
terms = sp.expand(sp.diff(inner0, X)).as_ordered_terms()
print("  expansion of d/dx[...]: %d terms (with u_t: %d)"
      % (len(terms), len(terms) + 1))
for t in terms:
    print("     ", t)

print("\nEq. (19), u_t + u_7x = 0:")
mu = sp.symbols("mu", positive=True)
u19 = sp.cosh(mu * (X - mu**6 * T))
report("u = cosh(mu(x - mu^6 t))", u19.diff(T) + u19.diff(X, 7))
print("  mu = 1.3 gives mu^7 = %.3f" % 1.3**7)

print("\nall exact" if ok else "\nSOMETHING IS NOT EXACT")
