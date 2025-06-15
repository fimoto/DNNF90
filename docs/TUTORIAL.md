# Tutorial: your own high-order differential equation

This walks through setting up a partial differential equation that is not
in the shipped set, from a blank directory to a verified solution. The
worked case is the two-dimensional heat equation, three variables and
second order, and it ships as `zwork/tutorial_heat2d` so every command
below can be run.

Everything shown was measured on the case as shipped; the numbers are
what you should see.

    cd zwork/tutorial_heat2d
    ../../build/serial.out          # or: sh job.sh

---

## 0. What the code will and will not do for you

It carries the value of a network and **every mixed partial derivative of
a chosen set** through one forward pass, and returns the weight gradient
of any loss you can write in terms of them. So:

* Your equation must be expressible as a residual `R = 0` built from
  derivatives of one scalar field — or, with the `System` block, a set
  of residuals over several field components (see below).
* The activation is `tanh` and the read-out is linear. The scalar forms
  carry a single output; a `System` block carries one component per
  output. These are not options: the derivative tables are the Bell
  polynomials of `tanh`, and the propagated derivatives are those of a
  linear read-out.
* The nonlinearity a scalar residual may contain is `u` times a
  derivative. In a `System` block a term may be a product of up to
  three factors, each a derivative of any component (`XUX`, `DXD`,
  `TRP`), which covers advection, flux divergences and body forces;
  general functions like `sin u` remain inexpressible. See the limits
  in section 8.

If your equation fits, the rest is bookkeeping, and this document is that
bookkeeping.

---

## 1. Write the residual as something that vanishes

Start from the equation and move everything to one side.

    u_t = a (u_xx + u_yy)          becomes      R = u_t - a u_xx - a u_yy

The sign convention is yours; only `R = 0` matters. Every term must be a
constant times a derivative of `u`, optionally times `u` itself, plus at
most one source term.

## 2. Fix the variable order, then read off the multi-indices

The input variables are the components of the first layer, in the order
you write them. Choose that order once and never change it:

    variable 1 = x
    variable 2 = y
    variable 3 = t

A **multi-index** counts how many times each variable is differentiated,
in that order:

| term | multi-index | why |
|---|---|---|
| `u` | `0 0 0` | no differentiation |
| `u_t` | `0 0 1` | once in variable 3 |
| `u_xx` | `2 0 0` | twice in variable 1 |
| `u_yy` | `0 2 0` | twice in variable 2 |
| `u_xyt` | `1 1 1` | once in each |

The order of the equation is the largest total, here 2.

**Which axis is time.** `Time_axis` defaults to the last variable, which
is why the order above puts `t` last. The setting matters only for the
`DXLAP` shorthand, which needs to know which axes the Laplacian runs
over; if you put time first, say `Time_axis 1 /` and the code will
reject a `DXLAP` that tries to differentiate along it.

## 3. The residual block

    Residual 3 /
    LIN   1.d0     0 0 1 /   u_t
    LIN  -0.05d0   2 0 0 /   -a u_xx
    LIN  -0.05d0   0 2 0 /   -a u_yy

`Residual m /` is followed by exactly `m` term lines. The four kinds are

| line | meaning |
|---|---|
| `LIN c a1..aD /` | `c` times the derivative of that multi-index |
| `UUX c a1..aD /` | `c` times `u` times that derivative; with `0..0` it is `c u^2` |
| `DXLAP c k [ix] /` | `c` times the `ix`-th derivative of the `k`-th power of the Laplacian, expanded for you. `ix` defaults to 1 |
| `SRC c /` | `c` times a source `f(x)` read per collocation point |

`DXLAP` is a shorthand for a long expansion: in three space variables
`DXLAP c 1 /` alone stands for three `LIN` lines, and `DXLAP c 3 /`
stands for ten. Write plain `LIN` lines when the operator is short, as
above; there is no Laplacian-only shorthand, so `u_xx + u_yy` is two
lines rather than one.

**Check what the parser understood.** A mistyped multi-index gives a
different equation, solved correctly and silently, so the run echoes the
operator before it starts:

    ### Residual as parsed (imposed as R = 0 at every collocation point)
    ###   axes are written x1..x3, with axis 3 shown as t (Time_axis)
    ###   term      coefficient  expression                    slot  multi-index
    ###    1      1.000000E+00  u_t                               4    0  0  1
    ###    2     -5.000000E-02  u_x1x1                            5    2  0  0
    ###    3     -5.000000E-02  u_x2x2                            6    0  2  0

Read this against the equation you meant. It is the only check on the
operator that exists.

## 4. Choose the carried set

    Hod_K      2 /   the highest total order in the residual
    1.d0             lambda_0
    0.d0             lambda_1
    0.d0             lambda_2
    Hod_dense  0 /   carry the closure of the residual indices

`Hod_K K /` is followed by exactly `K+1` lines, the weights
`lambda_0..lambda_K`. They matter only for a `HOD_DATA` term, where
`lambda_p` weights the targets of order `p` and `lambda_0` weights the
value; for a pure collocation problem set `lambda_0` to 1 and the rest to
0. **These `K+1` lines belong to `Hod_K`**: no other keyword may come
between them.

`Hod_dense 0` carries the **downward closure** of the multi-indices the
residual mentions instead of every index up to order `K`. Here that is 6
indices instead of 10:

    tools/alpha_order.py --d0 3 --k 2 --count      ->  10   (dense)
    grep -v '^#' hod_alpha_order.dat | wc -l       ->   6   (closure)

The cost of a point is proportional to the number carried, so the closure
is free speed. `docs/BENCHMARKS.md` explains why the set has to be
downward closed and how its size is counted.

## 5. Supply the points

Two kinds of term, two kinds of file. Records are read with
list-directed input, so **a record is numbers only**: comment lines are
not skipped.

**Where the equation is imposed** — a `COLLOCATION` term, one interior
point per record. With `Exact_solution 1` a last column holds the known
solution, reported on but never trained against:

    Loss_term  COLLOCATION  600  colloc.dat  1.d-1 /
    Exact_solution 1 /

    x                       y                       t                       u*
    1.77340027734810e-01   1.52821472123110e-01   2.35339868518061e-02   2.38608069820826e-01

**What pins the solution down** — a `DATA` term. A residual alone has
many solutions, so the initial condition and the boundary values go in
here as ordinary supervised points, `x(1:D0)` then `u`:

    Loss_term  DATA  440  data.dat  1.d0 /

This case uses 120 points at `t = 0` and the four walls at four times.
There is no separate keyword for boundary conditions: they are data.

**The two weights are the balance you tune.** `1.d0` on the data and
`1.d-1` on the residual worked here. If the fit satisfies the equation
but ignores the boundary, raise the data weight; if it matches the
boundary and violates the equation, raise the residual weight.

## 6. Run it, and make the run check itself

    Hod_check 1 /

turns on the in-run verification: it compares the gradient the run is
about to use against central differences of the loss, on your data and
your operator, and refuses to train if they disagree.

    ### [dW] weight-gradient FD check: 43 weights, max relative error = 0.84E-10
    ### [dW-R] PINN residual-gradient FD check: 43 weights, max relative error = 0.31E-11
    ### HOD self-check: ALL PASSED

Leave it on while you are setting a problem up. If it fails, the fault is
in the code, not in your input, and it should be reported.

## 7. Read the result

    grep -v '^#' history_ep0000000.dat | tail -1

Column 1 is the epoch, 2 the patience counter, 3 the training cost and 4
the validation cost, all per point. This case ends at
`cost_train = 4.12e-4`, `cost_val = 6.15e-4`.

`output_set0002.dat` is the collocation term: `x(1:D0)`, the prediction,
then the exact solution. Because the last two columns are prediction and
reference, the quality is one line of arithmetic:

    awk '!/^#/{d=$(NF-1)-$NF; s+=d*d; m+=$NF; n++; y[n]=$NF}
         END{m/=n; for(i=1;i<=n;i++) v+=(y[i]-m)^2; printf "R2 = %.6f\n", 1-s/v}' \
        output_set0002.dat

For this case, `R2 = 0.996480`, root-mean-square error `1.6e-2`, largest
error `4.3e-2`. **Always run this.** A network stuck at the mean of the
data still writes a smooth, plausible output file; the coefficient tells
you whether it learned anything.

## 8. When it does not work

| symptom | cause | what to do |
|---|---|---|
| `Residual: unknown term type` | a term line is not `LIN`, `UUX`, `DXLAP` or `SRC` | check the spelling |
| `init_hod_tables: seed multi-index exceeds K` | a residual term has higher total order than `Hod_K` | raise `Hod_K`, or omit it and let the residual set it |
| `read_data: record n of file accumulates M values where ... are required` | a record has the wrong number of columns | count them; remember the optional source and exact-solution columns |
| `input WARNING: unrecognized keyword ignored` | a misspelled keyword, silently doing nothing | fix the spelling; the run continues with the default |
| the echo shows a term you did not write | a mistyped multi-index | compare the echo against your equation |
| `R2` near 0 with a small cost | the fit is at the mean of the data | the residual weight is probably far too large |
| `GD_param: ... must be positive` | an adaptive rule was given a zero regularizer | give `ADAM` a positive fourth parameter |

Two limits are worth knowing before you start:

* **The nonlinearity.** In the scalar `Residual` block, `UUX` covers `u`
  times a derivative, so Burgers, KdV and the Zakharov--Kuznetsov family
  are expressible, and `u^2` through a zero multi-index. In a `System`
  block, `DXD` covers products of two derivatives such as
  `grad(rho).grad(phi)` and `TRP` three factors, so the incompressible
  Navier-Stokes equations and an electrohydrodynamic coupling are
  expressible (see `zwork/pinn_kovasznay` and `zwork/pinn_ehd`).
  Transcendental functions of the field such as `exp u` are **not**, in
  either form.
* **The coefficients are constants.** A spatially varying coefficient
  `c(x)` cannot be written. A source term can be read per point, so an
  inhomogeneous equation is fine; a heterogeneous one is not.

## 9. Going faster on small problems

For a network of a few hundred weights, the extended Kalman filter is far
stronger per pass than a gradient rule:

    GD_method  KALMAN
    GD_param   1.d-3 0.995d0 0.9999d0 0.d0 0.d0 /
               ! p1 initial covariance, p2 forgetting factor, p3 its schedule

On the shipped KdV benchmark this reaches a training cost of `2.6e-13` in
1000 epochs where the natural gradient reaches `1.5e-5` in 3000. The
dense covariance is quadratic in the number of weights, so the run
reports its size on startup and refuses above 2 GB; beyond that size add

    Kalman_mode DECOUPLED /

which keeps one covariance block per neuron (about 90x cheaper per epoch
at 2700 weights) at the price of roughly ten times the epochs per
accuracy decade -- still far ahead of a gradient rule in this regime.
Two cautions from the measurements: on coupled systems with source terms
the filter diverges under the default forgetting factor (fix `p2 = p3 =
1` to hold a state; see `docs/INPUT_KEYWORDS.md` under `Kalman_gate`),
and for such systems the full-batch L-BFGS or the dual natural gradient
of `zwork/pinn_ehd/input_cold*.dat` are the measured recommendations.

## 10. Where to look next

* `docs/INPUT_KEYWORDS.md` — every keyword, its default and its validation
* `docs/BENCHMARKS.md` — the record layout of every file, and the shipped cases
* `docs/DESIGN.md` — why the library carries mechanisms and the trainer carries policy
* `docs/USAGE.md` — builds, restarts, and the verification targets
  `make fdcheck.out`, `make harden`, `make negtests`

If your problem does not fit the residual language, the library
underneath does not have that restriction: it hands you every carried
derivative and takes the seed `dL/dT` of any loss you can differentiate.
`tools/example_customloss.f90` trains a Huber loss with a Sobolev penalty
in three lines of seed code, which the keyword menu cannot express.
