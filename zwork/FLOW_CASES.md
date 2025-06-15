# The flow cases

`pinn_kovasznay`, `pinn_cavity`, `pinn_ehd` and `pinn_ehd_mild` are the
coupled-system cases: several field components, a `System` block of
residuals, and (for two of them) an exact answer to compare against.
Each directory carries a `plot.gp` that draws the exact solution, the
learning curve and the network solution; run it with `gnuplot plot.gp`
after a run.

## Cold starts work under full-batch L-BFGS

The statement that these systems cannot be trained from a random start
was true of the minibatch first-order rules and is **not true of
full-batch L-BFGS**. The from-scratch Kovasznay run with the shipped
Adam input reaches R2 = -0.42/0.09/0.82; the same points under
L-BFGS descend monotonically. On the five-component EHD system the
cold start is a complete success (`zwork/pinn_ehd/input_cold.dat`,
`Lbfgs_m 40`, full batch):

    objective 0.974 -> 1.2e-5 per point, 6000 epochs, ~900 s
    R2 = 0.99997 0.99932 0.99985 0.99954 0.99759  (phi rho u v p)

— the same quality the two-stage protocol reaches, in one run with no
prior knowledge of the solution, which is the regime a real
application lives in. An Adam warmup before the L-BFGS buys nothing
(measured: 200 Adam + 300 L-BFGS epochs lands where 300 L-BFGS epochs
land). On the Kovasznay case, whose data lie on the boundary only,
the cold L-BFGS also descends steadily (1.50 -> 3.0e-3 in 500
epochs, R2 = 0.93/0.70/0.92 and still falling) but needs a longer run
than the EHD to reach the interior; the two-stage route below remains
the fast path there.

## Noisy data and wrong physics: the EnKF outer loop

For the regime a real application adds on top of the clean benchmark
-- noisy measurements, imperfect physics, inverse problems -- an
iPINNER-style outer loop (Lu, Mou & Lin, arXiv:2506.00731) is
implemented in `tools/ipinner_enkf.py` and measured on this case. The
ensemble Kalman filter runs on the PREDICTED FIELDS at the data
points, never on the weights, which is what makes it immune to the
sequential-scalarization failure the weight-space filters were shown
to have here; the paper's NSGA-III Pareto ensemble is replaced by
cheap cold-start L-BFGS members differing in `Rand_seed`. One
iteration: train M members on the noisy data, run the script (writes
`data_refined.dat` per member from the localized
perturbed-observation analysis), retrain each member from its
weights, repeat.

Measured, 10% rms Gaussian noise on `data.dat`:

- **Correct physics, 4 members**: the physics loss already filters
  the noise -- the ensemble prior at the data points is 2-3x closer
  to the truth than the observations, the gains are 0.07-0.14, and
  one iteration buys R2 improvements of only +0.0005..+0.0013 per
  component (ensemble mean 0.9998/0.9973/0.9982/0.9978/0.9883).
- **Wrong physics (viscosity 2x), 3 members, inflation 2**: the
  biased prior is where the loop pays. Every component improves in
  one iteration, most where the bias hurt most:
  v 0.9914 -> 0.9927, p 0.9779 -> 0.9805 (a 12% error-variance
  reduction in p), ensemble mean
  0.9994/0.9964/0.9962/0.9927/0.9805.

The division of labor this establishes: full-batch L-BFGS solves the
clean coupled problem outright; the EnKF outer loop is the tool to
reach for when the data are noisy and the physics is not fully
trusted, and its cost is M ordinary trainings per iteration.

## The two-stage procedure

For optimizer studies, and for cases like the Kovasznay where the
data constrain only the boundary, the two-stage procedure remains the
sharper instrument:

1. **Fit.** Train supervised on the exact (or manufactured) solution
   over the collocation cloud, with the residual weight tiny. This is
   `input_fit.dat`; its result is `fitted_weight.dat`.
2. **Train.** Restart from the fit with the boundary data and the
   residual together, full batch. What is being tested is whether the
   physics loss *holds* a state that satisfies it, which is a much
   sharper question than whether an optimizer can find that state, and
   it is the question the optimizer comparison in the papers needs
   answered first.

The shipped `fitted_weight.dat` is the stage-1 result, so stage 2 can
be run directly.

## Verifying pinn_kovasznay from a clean unpack

    tar xzf DNNF90.tar.gz && cd DNNF90 && make
    cd zwork/pinn_kovasznay
    cp fitted_weight.dat nn_weight.dat

Edit `input_nn.dat`: `Restart 1`, `GD_method LBFGS` with
`GD_param 0. 0. 0. 0. 0. /` (L-BFGS takes no learning rate), and
`Num_batch 3280` (L-BFGS is full-batch: 3480 points minus 200
validation; the input check states the required value if it is wrong).
Then

    ../../build/serial.out

Measured on a clean unpack (gfortran -O3, 2026-08-03): the L-BFGS
objective descends monotonically from the restart, and after 400
epochs (about five minutes)

    u  R2 = 0.99960
    v  R2 = 0.99807
    p  R2 = 0.99982

against the exact solution, computed from `output_set0002.dat` as
R2 = 1 - mean((pred-exact)^2)/var(exact) per component. Long runs can
fork in the last digits between builds (the verification notes cover
why), so the R2 level, not its fourth decimal, is the regression
signal. Anything that breaks this case at that level is a real
regression: the term table of the residual
is verified against the exact solution to 1e-15 by an independent
reader (`bench/post/check_kovasznay.py`).

To regenerate `fitted_weight.dat` itself:

    cp input_fit.dat input_nn.dat     # keep a copy of the original first
    ../../build/serial.out            # ~3.5 min, 20000 epochs of Adam
    cp nn_weight.dat fitted_weight.dat

The fit reaches R2 = 0.9992, 0.9995, 0.9986 (u, v, p).

## pinn_lax

A scalar case, included here because it needs the `System` term
products: Lax's fifth-order equation

    u_t + 30 u^2 u_x + 20 u_x u_xx + 10 u u_xxx + u_xxxxx = 0

carries a three-factor term (`TRP`: 30 u u u_x) and a product of two
derivatives (`DXD`: 20 u_x u_xx) that the scalar `Residual` block
cannot write. The exact soliton u = 2k^2 sech^2(k(x - 16 k^4 t)),
k = 1/2, is verified against the equation in 40-digit arithmetic;
`bench/post/check_lax_input.py` reads the term table back
independently (sympy-exact derivatives) and leaves 8e-16 at that
solution. `gen_data.py` regenerates `data.dat` (100 boundary/initial
points) and `colloc.dat` (200 interior points, exact solution in the
last column) with a fixed seed.

Cold start, full-batch L-BFGS (`Lbfgs_m 40`, 3000 epochs, ~106 s on
one core): objective 6e-10 per point, and against the soliton

    relative L2 = 1.0e-3    R2 = 0.999998

on the collocation set. `bench/post/make_fig_lax.py` draws the
profile and error-field figure from `nn_weight.dat`.

## pinn_lax7

The seventh-order member of the Lax hierarchy, written as Chen et al.
(Sci. Rep. 14 (2024) 23874) write it:

    u_t + d/dx[ 35u^4 + 70(u^2 u_xx + u u_x^2)
              + 7(2u u_xxxx + 3u_xx^2 + 4u_x u_xxx) + u_xxxxxx ] = 0

Expanded, nine terms, one of which -- `140 u^3 u_x` -- is a product of
FOUR factors and needs `QAD`; the others exercise `TRP`, `DXD`, `XUX`
and `TRM`. Exact solitary wave u = (1/2) sech^2((x-t)/2), verified in
40-digit arithmetic; `check_lax_input.py` reads the nine-term table
back and leaves 1.1e-14 at that solution.

Trained like the fifth-order case (100 boundary and initial points,
200 collocation points, 2-16-16-1) by cold full-batch L-BFGS
(`Lbfgs_m 40`, 3000 epochs, ~40 s on one core): objective 3.3e-10 per
point, and against the exact wave

    relative L2 = 8.8e-4    R2 = 0.999999

`Hod_check 1` on this case gives [dW] = 2.7e-10, which is the
finite-difference check of the four-factor adjoint. `gen_data.py`
regenerates the data with a fixed seed.

For scale, Chen et al. report 3.29e-3 (tanh) and 1.10e-3 (sine) on
this equation; their network, sampling and hardware differ, so it is
context and not a controlled comparison.

A warning from building this case: the `System nres nterm` line must
count EVERY term line. Writing nine term lines after `System 1 8`
would leave the last one out of the operator, so the parser stops on a
term line that falls outside its block rather than solving a different
equation.

## pinn_cavity

The lid-driven cavity at Re = 100 has no closed-form solution; the
reference is the Ghia, Ghia & Shin (1982) centerline profiles, and the
comparison is `bench/post/cavity_compare.py`. At the shipped size and
training length the run reproduces the shape of the profiles but not
their accuracy. `colloc.dat` carries coordinates only (nothing to
compare against pointwise), so `plot.gp` draws the network fields and
the learning curve, and the profile comparison is the python script's
job.

## pinn_ehd and pinn_ehd_mild

`pinn_ehd/input_nn.dat` is a copy of `input_cold.dat`, the full-batch
L-BFGS cold start, so `sh job.sh` reproduces the result reported in the
paper. The diverging Kalman configuration this section characterizes
ships beside it as `input_kalman_divergence.dat`; copy it over
`input_nn.dat` to reproduce the divergence.

The five-component electrohydrodynamic system with a manufactured
solution (`phi = sin x sin y`, `rho = cos x cos y + 1/2`, ...). The term
table is verified. The two-stage procedure applies here too:
`input_fit.dat` fits the manufactured solution (`fit_data.dat` is
columns 1,2,8-12 of `colloc.dat`; 15000 epochs of Adam, ~100 s) to

    phi 0.9996   rho 0.9990   u 0.9997   v 0.9992   p 0.9977   (R2)

and its result ships as `fitted_weight.dat`.

From that fitted start, on the composite objective the optimizer
descends (per point, `Loss_term` weights applied, which is also what
columns 3-5 of the history report):

- **L-BFGS** (full batch, `Num_batch 1420`): 2.64e-3 -> 4.2e-4 in 300
  epochs (~6x), monotone. Still the strongest rule here by a wide
  margin.
- **Adam** descends at eta = 2e-4 (2.80e-3 -> 1.47e-3 in 3000 epochs,
  ~2x) and is thrown out of the basin at eta = 2e-3 (jumps to 4e-2,
  recovers only to 7.5e-3). The step size is what decides it: small
  enough, and Adam holds and improves the fitted state, only far more
  slowly than L-BFGS.
- **The Kalman filter diverges** from the fitted start
  (2.8e-3 -> 1.44 in 50 epochs) at the shipped forgetting setting.
  `Sys_rnoise 1 30 30 30 1` slows
  the divergence ~5x; the innovation gate (`Kalman_gate`) never trips
  on it, which is diagnostic: the divergence is not driven by
  statistically outlying innovations. **The mechanism is the
  forgetting factor**: lambda = 0.995 divides the covariance by
  lambda at each of ~2700 updates per epoch, so the gains never decay
  and the filter random-walks off the fitted state along the
  linearization error of the nonlinear residuals. With **no
  forgetting** (`GD_param p1 1.d0 1.d0`, i.e. lambda fixed at 1) the
  gains decay as in recursive least squares and the walk stops:
  p0 = 1e-3 drifts only to 5.0e-3 in 50 epochs, and **p0 = 1e-5
  descends** (2.797e-3 -> 2.774e-3). p0 = 1e-4 sits on the boundary
  (dips, then drifts up). The recipe: forgetting for cold starts,
  lambda = 1 with a small p0 to refine a fitted state. For a **cold
  start on this system**, forgetting must be brief: `GD_param 1.d-3
  0.995d0 0.99d0` (lambda reaches 1 within ~0.2 epoch) descends
  monotonically 1.08 -> 0.62 in 40 epochs where the shipped
  `0.995/0.9999` oscillates without converging, and a plain
  `Restart 1` (which resets the covariance, since the filter state is
  not persisted) resumes the plateaued descent. Holding the
  forgetting for ~1 epoch (`0.99/0.999`) dips and then diverges. Even
  so, Adam reaches a lower cold-start cost in the same 80 epochs at
  ~1/360 of the wall time, so on this system the filter's per-epoch
  advantage does not hold; on the 2-D Poisson pair it does, by 67x in
  epochs and 4.5 orders in cost.
- The rest of the first-order sweep from the fitted start, per-point
  objective after 3000 epochs (start 2.80e-3): **RMSprop** eta=2e-4
  reaches 1.59e-3 and **AdaGrad** eta=1e-3 reaches 1.78e-3 (both
  descend, like Adam); **AdaDelta** overshoots and does not return
  below the start (8.9e-3 at lr=0.05; NaN at lr=1). The **natural
  gradient** jumps to 0.125 at the first event even with `Ngd_trust`
  and is impractically slow at 2741 weights (dense metric). None
  approaches L-BFGS (4.2e-4 in a tenth of the epochs).
- **`Sys_balance` measured, no benefit at this setting** (cyc=100,
  alpha=0.1, on Adam eta=2e-4): the residual weights drift to
  (1, 5.0, 8.1, 7.5, 15.2), the objective they define ends where it
  started (2.82e-3), and plain Adam without it reaches 1.47e-3. A
  caveat it exposed: rebalancing rescales the objective mid-run, so
  the history becomes a moving target while it is active.

`colloc.dat` carries `x y`, five sources (one per residual,
`Sys_src`), then the five exact components (`Exact_solution 1`) — 12
columns. `data.dat` carries `x y` and the five components — 7 columns.
