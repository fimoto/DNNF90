# Verification

What this distribution checks about itself, how to run each check, and
the numbers a clean build produces. Every figure below is measured on an
unpacked release with `gfortran -O3`; none of it is a reading of the
code.

## The checks

| Check | Command | Result |
|---|---|---|
| carried derivatives against an independent implementation | `make negtests` | 3.3e-19 at `K = 7`, all 330 slots |
| weight gradient at one point against central differences | `make fdcheck.out && ./build/fdcheck.out` | 2.2e-10 (`point adjoint vs FD`) |
| weight gradient over a batch against central differences | `make fdcheck.out && ./build/fdcheck.out` | 1.4e-11 (`batch adjoint vs FD`) |
| input derivatives of order <= 2 against central differences | `Hod_check 1` in any case, e.g. `zwork/hod_4d_k7` | 5.1e-9 to 1.2e-8 over the cases of the paper (`[dX]` line) |
| input-rejection contract and exact properties | `make negtests` | 25 of 25 |
| multi-output forward, first derivatives of every output | `make multiout_example.out` | < 1e-9, and component 1 identical to `net_eval_hod` to the last bit |
| multi-output adjoint, weight gradient | `make multiout_adj_example.out` | < 1e-9 |
| product terms, one to four factors, including a repeated factor and one term whose four factors are the same entry | `make negtests` (or `make product_adj_example.out`) | < 1e-10 |
| L-BFGS two-loop recursion against the Newton direction | `make lbfgs_check_example.out` | < 1e-14 |
| the linear seventh-order activation table against five seeds, four activations, the per-seed values shipped | `sh zwork/pinn_cosh7/run_seed_study.sh` (about ten minutes) | the medians and ranges of the table; only $J_0$ beats tanh on every seed |
| JAX Taylor mode on the directional task, same loss and same network as the Fortran column | `python3 bench/post/compare_frameworks.py --setting dir --kmax K` | 0.098, 0.130, 0.169, 0.205 ms at K = 1 to 4; out of memory beyond |
| the seventh-order activation table against eight seeds, four activations, the per-seed values shipped | `sh zwork/hod_4d_k7_order7/run_seed_study.sh` (about half an hour) | the medians and ranges of the table; each derivative-bounded activation beats tanh on 8 of 8 seeds |
| a C work space built for a network that was then freed, with the slot reused by a wider network | `make negtests` (or `make c_handles.out`) | `dnnf90_eval` returns -1; without the generation counter the same sequence aborts in `free()` |
| one process runs init/free/init across different configurations, in both directions: a five-residual System case (D0=3, K=2) and a plain DATA case (D0=1, K=1) | `make negtests` (or `make lifecycle_switch_example.out`) | 10, 1, 10 carried slots; `sys_nterm` and `sys_nres` are zero after every free |
| the weight-file header is checked on restart, on PREDICT and by the committee: activation, layer count and every layer width | `make negtests`; or train with `Activation BESSEL` and restart under TANH | the run stops naming the two activation codes; a changed layer width stops naming the layer |
| `Output_deriv 1` on a plain `DATA` case, where the derivative tables are empty until the network is synchronized | run `zwork/x2+1` with `Output_deriv 1` under `make harden` | the derivative file is written and no bound is violated |
| the geodesic correction vanishes on a model that is linear in the weights, for a loss weight of 1 and of 0.37 | a single-layer network (`Nlayer 2`) with `Ngd_geo`, one `NATURAL_GRAD` epoch | correction/step = 1.9e-14 and 2.1e-14; the same test on a 1-6-1 network gives 0.19, so the check does detect curvature |
| the natural-gradient step is unchanged when every residual of the objective is duplicated | copy `zwork/pinn_taylorgreen` with the `DATA` term removed, duplicate the 14 `System` terms onto residuals 4-6, run one `NATURAL_GRAD` epoch of each and compare `nn_weight.dat` | 8.8e-17 relative; the metric, the gradient and the trace damping all scale together, so the step does not |
| the static and shared libraries and the C interface | `make lib`, `make shared`, `make c_example.out`, then run it in `bench/kdv` | the example loads the trained network, prints value and derivatives, and reports a KdV residual of order 1e-4 |
| the `BLAS=1` build against the default loop kernels, for TANH, SIN, ERF, BESSEL and BESSEL1 | `make BLAS=1`, run a `DATA` case with each activation | the trained cost agrees to 6e-11 relative or better; `make BLAS=1 negtests` passes |
| the MPI build on 2, 3 and 4 ranks with a two-term loss, where the partition cuts across the terms, at a batch size that fits one rank's share of the smaller term | `make mpi`, then `mpirun -np N build/mpi.out` in `bench/kdv` | runs to completion; the minibatch is stratified inside each term, so no rank is left without points of a term |
| every shipped case under a hardened build (`-O0 -fcheck=all -finit-real=snan -finit-integer=-99999999 -ffpe-trap=invalid,zero,overflow`) | `make harden`, then run the cases | no bounds violation, no uninitialized read, no invalid or overflowing operation |
| the K=7 golden file against an independent multivariate Taylor jet in Python, which shares no code with the library | `python3 tools/make_hod_golden.py zwork/hod_4d_k7` | 2.2e-19 over all 330 slots (2.8e-17 over the 35 of `zwork/hod_4d_k3`) |
| `RMSPROP_NESTEROV` against a hand-computed three-step reference on a constant gradient (eta 0.1, rho 0.9, alpha 0.5) | drive `optimization_driver` directly | -0.316228, -0.387530, -0.385859 |
| the EHD term table, its sources and the shipped data against one manufactured solution, read back by an independent checker | build and run `tools/example_ehd_check.f90` | 1.1e-16 over all 25 terms |
| `dnnf90_init` -> `dnnf90_free` -> `dnnf90_init` in one process, and `dnnf90_nderiv` before and after the first evaluation | embed the library and call the cycle | the counts agree and the second init succeeds |
| the batched (BLAS) value path against the per-point path, for TANH, SIN, ERF, BESSEL and BESSEL1 | `make negtests` (or `make batch_act_example.out`) | < 1e-8 relative on the weight gradient |
| the weight file records the activation it was trained with, and a reader that would evaluate it with different analytics is refused | train any case with `Activation BESSEL`, then load `nn_weight.dat` against a TANH table set | `func 3` in the header; `net_load` stops |
| L-BFGS full-batch cost and gradient cover HOD_DATA | run any `HOD_DATA` case with `GD_method LBFGS` | non-zero cost that the line search reduces (`zwork/tour_sobolev` with `LBFGS`: 0.240 to 7.0e-5 in 300 epochs) |
| library path against the trainer, and the Kalman filter against the regularized normal equations | `make libverify_example.out` | machine precision |

## What is covered, and what is not

The table above lists checks; this one lists combinations, so that an
untested one is visible as a blank rather than hidden inside the word
"verified".  A figure is the value that combination actually prints;
a dash means the combination is not exercised by any shipped check.

| combination | forward / input FD | weight-gradient FD | notes |
|---|---|---|---|
| TANH, scalar, closure, HOD_DATA | 1.1e-8 | 5.0e-11 | `zwork/hod_4d_k3` |
| SIN, scalar, closure, HOD_DATA | 3.2e-9 | 4.0e-11 | same case, `Activation SIN` |
| ERF, scalar, closure, HOD_DATA | 5.8e-9 | 4.7e-11 | same case, `Activation ERF` |
| BESSEL, scalar, closure, HOD_DATA | 4.1e-7 | 5.2e-10 | same case, `Activation BESSEL` |
| BESSEL1, scalar, closure, HOD_DATA | 1.7e-8 | 3.4e-11 | same case, `Activation BESSEL1` |
| TANH, scalar, closure, PINN | 5.1e-9 to 1.2e-8 | 8.1e-10 to 1.4e-9 | the six cases of Table 1 |
| TANH, scalar, dense, PINN | --- | --- | `bench/scal_dense` runs it, and its trajectory is bitwise that of the closure run, but no separate FD figure is recorded |
| TANH, system (multi-output), closure, PINN | --- | ~3e-10 | `tools/example_ehd_grad.f90` (random points, so the figure moves within an order); the term table itself is checked to 1.1e-16 |
| every activation, batched (BLAS) value path | --- | < 1e-8 against the per-point path | `make batch_act_example.out` |
| non-TANH with the golden regression | --- | --- | the golden file is for TANH; other activations skip it by design |
| non-TANH weight file, save and load | --- | --- | the activation code is recorded and a mismatched reader is refused, but no value round-trip is asserted |
| BLAS build of the full suite | --- | --- | not run here: the reference BLAS available in the test environment does not link |

The blanks are the honest state of the distribution, not an omission
from this file.

The rows with a stated figure --- the two `fdcheck` comparisons,
`negtests` and `libverify_example` --- are reproducible to the digit:
those checks run on fixed data, and `libverify_example` prints its own
pass thresholds beside each result. The `[dX]` row is a range because
it collects the value printed by each case of the paper, every case on
its own fixed data. The rows given as bounds draw their test points at
random on each run, so the figure
they print moves within about an order of magnitude from run to run;
the bounds held over repeated runs and are not values to be matched
exactly. All of them sit far
below the size of any error that would matter; a run that prints, say,
1e-6 for the product terms is a defect, one that prints 3e-11 instead
of 9e-12 is the same check passing.

`Hod_check 1` in any input file runs the two finite-difference
comparisons on that case, so a case of your own is checked the same way
as the shipped ones. `make harden` rebuilds with array-bounds checking,
signalling-NaN poisoning of uninitialized values and IEEE traps, for the
class of defect a numerical comparison cannot expose. Every benchmark
directory holds a reference log produced by the released code.

## Solved cases

Reference results for the shipped cases, from a clean unpack.
`docs/BENCHMARKS.md` describes the inputs and outputs of each;
`zwork/FLOW_CASES.md` covers the flow and coupled-system cases in
detail.

| Case | Input | Result |
|---|---|---|
| Kovasznay flow (two-stage) | `zwork/pinn_kovasznay` | R2 = 0.99960 / 0.99807 / 0.99982 |
| five-component EHD system, cold start | `zwork/pinn_ehd/input_cold.dat` | objective 0.974 to 1.2e-5 per point in 6000 epochs (about 900 s on one core), R2 = 1.00000, 0.99995, 0.99999, 0.99999, 0.99997 |
| the same, natural gradient on minibatches | `zwork/pinn_ehd/input_cold_ngd.dat` | objective 0.970 to 1.1e-6 per point in 520 epochs on 120-point batches, R2 = 1.00000 in every component |
| Lax's fifth-order equation (`System` product terms, term table read back independently to 8e-16) | `zwork/pinn_lax` | relative L2 = 1.0e-3, R2 = 0.999998, 3000 epochs (~106 s) |
| Lax's seventh-order equation (the `QAD` four-factor term; table read back to 1.1e-14, four-factor adjoint checked against central differences to 2.7e-10) | `zwork/pinn_lax7` | relative L2 = 8.8e-4, R2 = 0.999999, 3000 epochs (about 40 s on one core) |
| 2-D Poisson, extended Kalman filter | `zwork/pinn_poisson2d_kalman` | 5.5e-10 per point in 300 epochs |

## Choosing an optimizer

The menu is documented in `docs/INPUT_KEYWORDS.md`; this is what the
measurements support.

- **A coupled system whose collocation set fits in one batch**:
  full-batch L-BFGS with `Lbfgs_m 40`. It needs no step size and solves
  the shipped EHD system from random weights.
- **A collocation set too large for one batch**: the natural gradient
  with `Ngd_dual` and `Ngd_trust`, which builds one Gauss-Newton row per
  residual and solves through the Gram matrix. Its step size is the
  fragile parameter; `eta = 0.2` with `Ngd_trust 1.d-2` is the measured
  optimum on the EHD case and `0.5` diverges.
- **A small network on a weakly nonlinear observable**: the extended
  Kalman filter, which is far stronger per epoch in that regime.
  `Kalman_mode DECOUPLED` extends it past the dense covariance limit.
  On coupled systems carrying source terms the filter needs its
  forgetting factor fixed at one, and is still outrun by the two
  routes above.
- **Noisy data with physics that is not fully trusted**: the
  observation-space ensemble Kalman outer loop of
  `tools/ipinner_enkf.py`, over several ordinary trainings.

`Lbfgs_ss` (self-scaled BFGS), `Ngd_geo` (geodesic acceleration),
`Kalman_iter` (iterated EKF) and `Sys_balance` are implemented and
available; on the shipped cases none of them improves on the defaults,
and `docs/INPUT_KEYWORDS.md` gives the measured comparison for each.
