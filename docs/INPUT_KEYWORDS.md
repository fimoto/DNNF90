# Input keyword reference

Each line of `input_nn.dat` has the form `Keyword value ... /`; some
keywords are followed by a continuation block. Lines beginning with `#`
or `!` are comments. An unrecognized keyword produces a warning, so a
misspelling does not silently fall back to a default. If a keyword
appears more than once, the last occurrence wins.

## Network

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `Nlayer L` | required | Number of layers, followed by `L` lines giving the layer widths | all |
| `Activation NAME` | `TANH` | `TANH`, `SIN`, `ERF`, `BESSEL` (J_0) or `BESSEL1` (J_1). The adjoint reaches order K+1, so the activation must be C^(K+1); all five are. They differ in how fast their high derivatives grow, which is what the deep slots carry: at order eight the largest values are about 1220 for tanh, 454 for erf, exactly 1 for sin and 0.27 for J_0, whose derivatives are binomial combinations of Bessel functions and so are bounded by one at every order. `ERF` uses the Fortran 2008 intrinsic for its value, which is the one place the distribution reaches past Fortran 2003 and is excepted from `make f2003check`. The golden regression of `Hod_check` is generated for tanh and is skipped for the others; the finite-difference checks apply to all three | `zwork/hod_4d_k3` |
| `Init_w BESSEL_INIT` | - | Derived from the statistics of J_0 rather than copied from the periodic scheme: the variance of J_0(a) for a normal pre-activation is maximal at spread 2.75, which fixes the weight bound through Var[pre] = n b^2 E[z^2]/3. A third of that spread is used, the full value starting so deep in the oscillatory region that the initial loss is four orders larger. On the shipped high-order case it does not yet beat Glorot (1.0e-3 against 3.7e-4), so it is offered as a starting point rather than a recommendation | `zwork/hod_4d_k3` |
| `Init_w SIREN` | - | Places the pre-activation inside one oscillation, for the oscillatory activations `SIN` and `BESSEL` only. The first layer is drawn from U(-w0/fan_in, w0/fan_in) and every later layer from U(-sqrt(6/fan_in)/w0, +sqrt(6/fan_in)/w0), with w0 set by `Init_w_omega` (default 30). The optimum depends on the input range: on the shipped high-order case, whose inputs lie in [-1,1], w0 = 12 works and 30 does not | `zwork/hod_4d_k3` |
| `Init_w NAME` | required | Weight initialization: `Random`, `LeCun`, `Glorot`, `He`, each with `_unif` or `_norm` | `zwork/tour_derivfit` |

## Training data

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `Loss_term FORM npoints file weight [batch]` | at least one | One term of the composite loss, self-contained on one line. `FORM` is `DATA` (supervised values), `HOD_DATA` (values and all carried derivatives) or `COLLOCATION` (a cloud of points where the PDE residual is minimized; it carries no targets). `weight` multiplies the term in the total loss, so the data-to-residual balance of a physics-informed run is the ratio of two weights. The optional sixth field gives this term its own minibatch size; see the note below the table. The number of terms is counted, not declared. See `docs/DESIGN.md` | all |
| `Task NAME` | `TRAIN` | `TRAIN` fits the weights; `PREDICT` reads `nn_weight.dat`, writes the output files and exits without training; `COMMITTEE` evaluates an ensemble of already trained networks and exits | `zwork/tour_committee` |
| `Committee m` | - | `m` member weight files follow, one per line, and `m >= 2`. Used by `Task COMMITTEE`: the mean and the sample standard deviation of every carried slot are written per point to `output_committee_set*.dat`, so the spread of a derivative is reported as well as that of the value. With `Output_deriv 1` the dense first-order set is carried for exactly that purpose | `zwork/tour_committee` |
| `Num_validation n` | 0 | Validation points, in [0, N-1]. With 0 the patience test falls back to the training cost | `zwork/tour_schedule` |

### How the minibatch is drawn

The draw is stratified by `Loss_term`.  Each term contributes a fixed
number of points to every batch, drawn uniformly without replacement
from that term's own training points.

Under MPI the batch must also fit in one rank's share of every term:
with `nprocs` ranks a term of `n_j` training points offers about
`n_j/nprocs` candidates, and the draw stops if a term is asked for more
than that.  `Shuffle` has no effect under MPI for the same reason the
draw is stratified --- a global permutation would move points between
terms --- and the reader says so when it is set.

`Num_batch` must be at least the number of `Loss_term` lines, since
every term places at least one point in every batch.  Under MPI each
term is split across the ranks separately, so that every rank holds a
share of every term; splitting the pooled set instead would leave a
rank with none of one term and no stratified draw possible.

Without the optional sixth field, a term contributes its share of
`Num_batch` in proportion to the training points it kept,
`round(Num_batch * n_j / Ntrain)`, with the last term taking the
remainder so the sizes sum to `Num_batch` exactly.

With the field, the term draws exactly that many points and
`Num_batch` becomes their sum.  Either every `Loss_term` gives a size
or none does; a mixture is refused, since the two ways of deciding the
epoch size would contradict each other.  Asking for a whole set is the
intended use, so a size above the points the term kept is clamped to
them, with a message: the validation draw is over the pooled index
space, so a set of 100 points may keep 84 on one seed and 89 on
another.  A size above the term's total point count is a mistake and
stops the run.

Why stratify at all.  Drawing from the pooled training set instead
leaves the number of points of each term in the batch random.  With 93
boundary points, 177 collocation points and a batch of twenty, the
boundary count is hypergeometric with mean 6.9 and standard deviation
2.0, and it falls to two or fewer on 1.1% of the epochs (to zero on
0.014%).  On those steps the boundary condition all but disappears
from the gradient.  The pooled gradient is still unbiased, but the
objective each step sees is not the same objective, which is the
defect the stratified draw removes.  It is worth being plain that on
this benchmark the removal is not measurable: stratified and pooled
draws of the same size converge alike, within the spread over seeds.
The defect is real, its cost here is not.

Which to use.  The proportional default is the better choice unless
the terms are very unequal in size.  On the KdV benchmark of the paper
(93 boundary, 177 collocation), giving the boundary term all its
points and the collocation term twenty reaches 8.9e-8 to 2.3e-7 over
three seeds, while the proportional draw of the same total size
reaches 3.9e-8 to 5.1e-8: spending the budget on the boundary term
starves the residual term, whose sampling noise then dominates.  The
per-term field earns its place when the boundary set is small next to
the collocation set --- the common arrangement in the PINN literature,
where a few dozen boundary points sit beside thousands of collocation
points and the boundary term is cheap to take in full.

## Optimization

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `GD_method NAME` | `SIMPLE` | `SIMPLE`, `SIMPLE_SCHEDULE`, `SIMPLE_GMAXCLIP`, `MOMENTUM`, `NESTEROV`, `ADAGRAD`, `RMSPROP`, `RMSPROP_NESTEROV`, `ADADELTA`, `ADAM`, `NATURAL_GRAD`, `KALMAN` | `bench/opt_*` |
| `GD_param p1..p5` | required | Method hyper-parameters on their own line, in any order relative to `GD_method`; for `SIMPLE_SCHEDULE`, eta(t) = p1/sqrt(t+1) + p2. All-zero values are rejected | `zwork/tour_schedule` |
| `GD_method KALMAN` | - | The per-pattern extended Kalman filter, the training method of the n2p2 tool chain, rather than a gradient method. One update presents one scalar observable: the value for a `DATA` point, each carried derivative of non-zero lambda for a `HOD_DATA` point, and the residual against a target of zero for a `COLLOCATION` point. `GD_param` becomes `p1` the initial diagonal of the covariance, `p2` the initial forgetting factor, `p3` its schedule. The covariance is n_w by n_w, so this is a method for small networks: the run reports its size and refuses above 2 GB | `zwork/pinn_poisson2d_kalman` |
| `Num_batch n` | min(10,N) | Minibatch size, in [1, Ntrain]. Use the full set for timing comparisons. Ignored, and reported as such, when every `Loss_term` gives its own size: the epoch then draws their sum | all |
| `Epoch n` | 1000 | Number of epochs | all |
| `NGD_schedule_eta NAME a b` | `NONE` | Natural-gradient step schedule: `NONE`, `SIMPLE`, `EXP` | `bench/opt_ngd` |
| `NGD_schedule_mu NAME a b` | `NONE` | Damping schedule for G = (1/N)[sum j j^T + mu tr(.) I] | `bench/opt_ngd` |
| `NGD_eta_bound VALUE` | `0` | Floor under the scheduled step: the schedule above may decay the step indefinitely, and this stops it, so a long run does not stall at a step that no longer moves the weights. Only meaningful with a decaying `NGD_schedule_eta` | `bench/opt_ngd` |
| `NGD_mu_bound VALUE` | `0` | Floor under the scheduled damping, the counterpart of `NGD_eta_bound`. Keeping a small positive floor keeps the damped metric safely invertible once the schedule has decayed mu; on a coupled system, where the metric is closest to singular, this is the safer setting | `bench/opt_ngd` |
| `GD_method KALMAN` weighting | - | The filter balances observations through its observation noise and forgetting factor, not through the loss weights. A `Loss_term` weight and the HOD `lambda` select which observations are presented but do not scale them; multi-output `DATA`/`HOD_DATA` fitting is refused, because the observation is single-output | `bench/opt_kalman` |

## Schedule, stopping and output

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `Validation_cyc n` | 1 | Validation interval in epochs | `zwork/tour_schedule` |
| `P_max n` | unlimited | Patience limit: training stops after this many *consecutive* validation cycles without improvement, and any improvement resets the counter | `zwork/tour_schedule` |
| `Conv x` | 0 | Stop when the training cost per point falls below x | `zwork/tour_schedule` |
| `Checkpoint_cyc n` | 1000 | Interval, in epochs, of the `checkpoint_ep*.dat` files | all |
| `Output_deriv n` | 0 | 1 writes dN/dx to `output_deriv.dat` after training | `zwork/tour_derivfit` |
| `Restart n` | required | 1 restarts from `nn_weight.dat`. Missing optimizer-state files give a weight-only cold start; schedules continue on the absolute epoch counter | - |
| `Rand_seed n` | 0 | Positive values make a run reproducible; 0 seeds from the clock | `zwork/tour_*` |

## High-order derivatives and PINN

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `Hod_k K` | 0 | Highest order carried, followed by `K+1` lines giving the weights lambda_0 ... lambda_K. Setting a high-order target to zero with a small weight makes the loss a Sobolev-type derivative penalty | `zwork/hod_4d_*`, `zwork/tour_sobolev` |
| `Hod_check n` | 0 | 1 runs the finite-difference self-check | `zwork/hod_4d_*` |
| `Hod_dense n` | 0 | 1 carries all multi-indices; 0 carries the closure of the seeds | `zwork/hod_4d_*` |
| `Hod_alpha_file f` | none | External list of seed multi-indices | `zwork/hod_4d_k7_active` |
| `Residual m` | - | Residual definition: `m` term lines follow. A term is `LIN c a1..aD /` (c times a derivative), `UUX c a1..aD /` (c times u times a derivative), `DXLAP c k /` (divergence form of the k-th Laplacian power), or `SRC c /` (a source) | `bench/kdv`, `bench/zk7` |
| `Time_axis n` | last axis | Which input axis is time. `DXLAP` needs it: its Laplacian runs over the axes that are not time. The default reproduces the convention the shipped cases follow, and the keyword may appear anywhere in the file, since the `DXLAP` terms are expanded only after the whole input has been read | `bench/zk7` uses the default |
| `DXLAP c k [ix]` | - | A `Residual` term line: `c` times the `ix`-th derivative of the `k`-th Laplacian power, expanded into ordinary multi-index terms. `ix` defaults to axis 1. Giving the time axis as `ix` is rejected | `bench/zk7` |
| `SRC c` | - | A `Residual` term line: the source `c*f(x)`, with `f` read per collocation point from an extra column of the collocation file placed right after the coordinates. This is what makes an operator inhomogeneous, for example Poisson written as `u_xx + u_yy - f = 0`. It enters the residual additively, so it does not appear in the seed | `zwork/pinn_poisson2d` |
| `Exact_solution n` | 0 | 1 compares against the exact solution supplied with the data | `zwork/pinn_*` |

## Coupled systems

The scalar `Residual` block writes one residual of one field. A system
of residuals over several field components — the incompressible
Navier-Stokes equations, a charge density coupled to a potential —
needs the `System` block instead. The network's output width is the
number of field components, and the training carries every derivative
of every component up to `Hod_k`.

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `System nres nterm /` | - | `nres` residuals defined by the `nterm` term lines that follow (at most 256 terms, `nres` at most 16). Each term is a product of up to four derivative factors, `c * d^d u_l * d^g u_k * d^b u_i * d^a u_j`, and the five term types are the cases of that: `TRM` one factor, `XUX` two with the first undifferentiated, `DXD` two both differentiated, `TRP` three, `QAD` four. With one component and `ic = jc = 1` the forms reduce to the scalar `LIN` and `UUX`, so older input still means what it did | `zwork/pinn_kovasznay`, `zwork/pinn_ehd` |
| `TRM ir jc a1..aD c /` | - | Term line: `c * d^a u_jc`. `ir` is the residual the term belongs to, `jc` the component that is differentiated, `a1..aD` the multi-index over the `D` input axes | `zwork/pinn_kovasznay` |
| `XUX ir ic jc a1..aD c /` | - | Term line: `c * u_ic * d^a u_jc`. What a coupled system needs and the scalar block cannot write: the advection `u u_x + v u_y`, the transport of a charge by a flow, the body force `rho E` | `zwork/pinn_kovasznay` |
| `DXD ir ic b1..bD jc a1..aD c /` | - | Term line: `c * d^b u_ic * d^a u_jc`, a product of two derivatives. The divergence of a flux needs it: `div(rho grad phi) = grad(rho).grad(phi) + rho lap(phi)`, and the first term is a product of two first derivatives of different components | `zwork/pinn_ehd` |
| `TRP ir kc g1..gD ic b1..bD jc a1..aD c /` | - | Term line: `c * d^g u_kc * d^b u_ic * d^a u_jc`, three factors, which a compressible momentum flux `rho u u_x` or a field-dependent mobility needs | - |
| `QAD ir lc d1..dD kc g1..gD ic b1..bD jc a1..aD c /` | - | Term line: `c * d^d u_lc * d^g u_kc * d^b u_ic * d^a u_jc`, four factors. The quartic terms of the higher dispersive hierarchies need it: the seventh-order Lax equation carries `140 u^3 u_x` | `zwork/pinn_lax7` |
| `Sys_src /` | off | Declares that the collocation file carries one source column per residual, placed between the coordinates and the exact solution. A homogeneous system omits it, and the file is then narrower; getting this wrong produces a record-length error, which is the check working | `zwork/pinn_ehd` |
| `Sys_wcomp w1..wm /` | all 1 | Weights each of the `m` field components in the supervised loss. Components of unequal size need it: an unweighted sum of squares is about whichever component is largest. Measuring `1/‖y_i‖²` from the data is the usual choice and is what the shipped cases do | `zwork/pinn_kovasznay` |
| `Sys_wres w1..wr /` | all 1 | Weights each of the `r` residuals in the collocation loss, for the same reason. Must appear after `System` (it reads `nres` values) | `zwork/pinn_ehd` |
| `Sys_rnoise r1..rr /` | all 1 | Gives each residual its own observation noise for the Kalman filter. It is exact rather than approximate: scaling both the observation row and the innovation by `1/sqrt(r)` reproduces a noise `r` on the unscaled quantities. A residual whose source is much larger than the others produces large innovations, and the filter, which trusts every observable equally, then spends its covariance on that one; raising its noise says the observation is less reliable. Must appear after `System` | - |
| `Kalman_gate g /` | 0 (off) | Innovation gate of the extended Kalman filter (a robust/adaptive-R EKF): an observation whose innovation exceeds `g` standard deviations of the filter's own predicted spread, `xi^2 > g^2 (lambda + j P j)`, has its observation noise inflated (row and innovation scaled together, the exact `Sys_rnoise` mechanism) until it does not, which bounds every rank-1 step. Gated updates are counted. Measured caution: the EHD divergence is *not* innovation-outlier-driven — the gate does not trip there — so this option protects against outliers but does not cure that case. What does: no forgetting. With `GD_param p2 = p3 = 1` (lambda fixed at 1) the gains decay as in recursive least squares and the filter holds a fitted state (`p1 = 1e-5` descends slowly; the shipped `0.995/0.9999` forgetting re-inflates the covariance ~2700 times per epoch and the filter walks away). Forgetting buys cold-start speed at the price of no stationary behavior | - |
| `Kalman_mode DENSE\|DECOUPLED /` | `DENSE` | `DECOUPLED` replaces the dense covariance with the node-wise block diagonal (one block per neuron over bias + fan-in, Puskorius-Feldkamp): the gain keeps the global innovation denominator, only the cross-neuron covariance is dropped. Memory and per-update work fall from `nw^2` to the sum of block sizes squared (~60x on the shipped EHD net; measured 87x wall-clock per epoch there), and the 2 GB dense guard no longer applies, so the filter reaches force-field widths. The price is the convergence rate: on the KdV bench it reaches 1.2e-7 in 0.5 s where the dense filter reaches 7e-14 in ~6 s, and on Poisson 2D 3.2e-8 in 5.4 s / 3000 epochs against 5.5e-10 in 6.5 s / 300 — roughly ten times the epochs per decade, at a tenth the cost per epoch. A separate measured benefit: the block-diagonal covariance strongly damps the forgetting-driven divergence on the EHD fitted start (drift to 2.3e-2 where the dense filter reaches 1.44), since the cross-neuron gain coupling was part of the driver | - |
| `Kalman_iter n /` | 1 | Iterated EKF (IEKF): each observation is relinearized `n` times at the running iterate before the covariance is updated once — Gauss-Newton on the single-observation MAP problem, the textbook remedy when the observation is nonlinear in the state. Measured verdict on the coupled EHD, and it is a negative one worth having: in the stable (brief-forgetting) regime n=3 is neutral (0.564 vs 0.560 at 200 cold epochs), and in the forgetting regime it *amplifies* the divergence (0.44 vs 0.023 from the fitted start), because solving each of thousands of mutually inconsistent scalar observations more exactly increases the inter-observation thrashing — the linearization error the IEKF removes was acting as a damper. The failure mode of per-pattern filtering on such systems is the sequential scalarization itself, whose batch cure is a joint Gauss-Newton step, i.e. the full-batch quasi-Newton route | - |
| `Kalman_q q /` | 0 (off) | Process-noise injection: `q` is added to the diagonal of P after every update. The continuous alternative to the forgetting factor — lambda < 1 inflates all of P and never lets the gains decay, q = 0 with lambda = 1 lets them decay to nothing — and to the manual covariance resets. Measured on the cold EHD (decoupled, lambda = 1): stable and equivalent to the fast lambda-to-1 schedule, no dramatic gain | - |
| `Sys_balance cyc alpha /` | off | Rebalances `Sys_wres` every `cyc` epochs so that each residual contributes a gradient of comparable size. The gradient norm of each residual's own loss term is measured on a sample of at most 40 training points, and each weight is moved a fraction `alpha` of the way towards `gmax/gnorm(ir)`; the running average keeps the weights from chasing the noise of one batch. The rebalanced weights are printed each time | - |

Data files of a system: `data.dat` is `x(1:D) y(1:m)` with `m` the
component count; `colloc.dat` is `x(1:D)`, then with `Sys_src` one
source per residual, then with `Exact_solution 1` the exact solution of
every component.

## L-BFGS

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `GD_method LBFGS` | - | A quasi-Newton update with a backtracking Armijo line search and no learning rate: `GD_param` may be omitted entirely (L-BFGS is the one method exempt from the all-zero rejection). It is a full-batch method — the curvature pairs `(s,y)` only mean anything if consecutive gradients are of the same function, and a resampled minibatch changes the function every step — so `Num_batch` must equal the training points, which is the file total minus `Num_validation`, and the input check says so if it does not. Its objective goes to `lbfgs_history.dat` (one row per epoch: epoch, accepted loss, step length, line-search trials, accepted flag), not to the training history, because the history is a different sum kept common to all update rules so they stay comparable | - |
| `Lbfgs_m n /` | 8 | Number of stored curvature pairs, in [1,200]. More memory buys a better quadratic model per epoch at O(n·n_w) extra work in the two-loop: on the cold-start EHD case, 300 epochs reach 6.3e-3 (m=8), 4.6e-3 (m=20), 3.3e-3 (m=40), and m=40 also wins at equal wall time | `zwork/pinn_ehd` (`input_cold.dat`) |
| `Lbfgs_expand n /` | 0 (off) | Forward-tracking line search: when the first trial step satisfies Armijo at once, try doubling it up to `n` times and keep the longest step that still satisfies the condition and lowers the loss. 0 reproduces the pure backtracking search exactly. Measured on the cold EHD (n=4): 19% lower cost at equal epochs, but the extra cost evaluations lose at equal wall time (127 s vs 83 s for 300 epochs) — an epoch-budget tool, not a wall-clock one, at that setting | - |
| `Lbfgs_ss /` | off | Self-scaled BFGS (SSBFGS, Oren-Luenberger): every stored pair gets its own curvature factor `tau_j = s_j.y_j / (y_j^T H_j y_j)` at its own level of the two-loop recursion, instead of the newest pair's `s.y/y.y` being applied once to the initial Hessian. `H_j` is the operator built from the older pairs, so the factors are filled oldest-first by a nested application of the same recursion (O(m^2 n) per direction, negligible beside one full-batch gradient). Verified: with one stored pair it reproduces plain L-BFGS to eight digits over fifty iterations. Measured verdict here: **slightly worse than plain L-BFGS**, consistently -- EHD cold start at 300 epochs 5.2e-3 vs 3.3e-3 (m=40), at 100 epochs 8.8e-2 vs 6.6e-2 (m=8), Kovasznay cold at 150 epochs 8.2e-2 vs 7.4e-2. A cheap variant approximating `H_j` by the identity is far worse still (6.7e-1 at m=8), which is worth knowing if the idea is revisited | - |
| `Lbfgs_wolfe c2 /` | 0 (off) | Add the curvature (second Wolfe) condition `g(w+td).d >= c2 g(w).d`, `c1 < c2 < 1`, to the line search: steps that decrease the loss but stop while the slope is still steep are rejected and the step is grown rather than shrunk, with proper bracketing. Costs a gradient instead of a cost evaluation per trial. Implemented because the self-scaled literature requires it; measured to change nothing on these problems, for the instructive reason that the first trial `t = 1` already satisfies both conditions almost always -- the line search was not what limited SSBFGS here | - |
| `Lbfgs_verbose /` | off | Prints the accepted and recomputed loss at each exit, which is how a mismatch between the search's function and the trainer's is caught | - |
| `Lbfgs_scan /` | off | Extra backtracks before the step is given up and the memory reset | - |

## Natural-gradient damping

| Keyword | Default | Meaning | Demo |
|---|---|---|---|
| `Ngd_damping TRACE\|ABS /` | `TRACE` | Chooses `G + mu tr(G) I` or `G + mu I`. The trace form is relative and its meaning changes with the weight count; the absolute form does not, which matters on the networks a coupled system needs | - |
| `Ngd_dual /` | off | Solve the natural-gradient step through the Gram (dual) matrix: the push-through identity `(G+cI)^-1 b = (1/c)[b - (1/N) J^T K^-1 J b]`, `K = cI + (1/N)JJ^T` (N_batch x N_batch), gives the identical step -- verified digit-for-digit against the primal solve on `bench/opt_ngd` -- at O(N^2 nw + N^3) instead of O(N nw^2 + nw^3). Measured 20.7x faster there (0.375 s vs 7.76 s / 600 epochs), and it is what makes NATURAL_GRAD runnable at coupled-system widths: on the 2741-weight EHD net the primal route was impractical, the dual route runs 600 epochs in 6 s (eta = 0.01 holds a fitted state; see the per-residual rows below, which is what makes it competitive with L-BFGS there) | `bench/opt_ngd` |
| (no keyword: automatic) | - | **Per-residual Gauss-Newton rows.** `NATURAL_GRAD` builds one metric row per RESIDUAL of a system (or per component of a multi-component fit), not one per point, with the exact `sqrt(w_r)` factorization of the weighted objective. A single row per point (`d(sum_r w_r R_r^2/2)/dw`) would make the outer product the empirical Fisher -- rank one per point, blind to how the residuals trade against each other; the per-residual rows make the metric the true `J^T J` of the residual vector. Measured on the EHD from the fitted start (`Ngd_dual`, `Ngd_trust 1e-2`, eta 0.2, 600 epochs): 2.76e-3 -> 4.6e-4, against 2.94e-3 with the old rows -- and cold from random weights, 0.743 -> 2.8e-4 in 900 epochs (~150 s) with R2 = 0.99993/0.99990/0.99981/0.99987/0.99865, i.e. the natural gradient now solves the coupled system outright, on minibatches, in the same league as full-batch L-BFGS. The run prints the row count at startup | `zwork/pinn_ehd` |
| `Ngd_geo h alpha /` | off | Geodesic acceleration (Transtrum-Sethna) of the natural-gradient step: one extra evaluation of every observable at `w + h*delta1` gives the second directional derivative, a second solve against the SAME metric (cheap under `Ngd_dual`) gives the correction `delta2`, and the step is `delta1 + delta2/2` when `|delta2|/2 <= alpha*|delta1|`, else plain. Eta-consistent (the difference is taken along the applied step). Scalar Gauss-Newton observables only (single-output DATA, scalar collocation); a system needs per-residual rows first and the code stops with that message. Now available for systems and multi-component fits too (each metric row carries its own observable). Honest status: still not a win here. On `bench/opt_ngd` it is neutral-to-slightly-negative; on the EHD system the spread across `h` (1.97e-3 at h=0.01 and h=0.3, 2.47e-3 at h=0.1, against 2.29e-3 without) is the size of the trajectory-to-trajectory variation of a minibatch run, so the effect is within noise while the cost is ~2x per epoch. Its literature home is full-batch Levenberg-Marquardt on stiff plateaus, which at this row count would need an N_row x N_row solve and is not reachable through the dual route | - |
| `Ngd_trust mu0 /` | off | Adapts an absolute damping from the observed change in the loss, Levenberg-Marquardt style: a step that lowered the loss divides `mu` by 3, one that raised it multiplies by 5, bounded to `[1e-6, 1e8]`. Setting it forces `Ngd_damping ABS` | - |

`Math_hod` and `Pinn` require `Hod_k` >= 1 and an identity output
layer. The scalar `Residual` and `Math_hod` forms carry one field, so
the output width must be 1; a `System` block lifts that, with one
component per output.
Negative lambda values are rejected.

## MPI

| Keyword | Default | Meaning |
|---|---|---|
| `Average_cyc_mpi n` | 1 | Interval in epochs between weight averages across ranks |
| `Shuffle n` | 0 | Accepted and ignored under MPI (the draw is stratified inside each rank's share, so a global permutation would move points between terms; the reader says so when it is set). In serial runs the minibatch draw already mixes the data |
| `Shuffle_cyc_mpi n` | 1000000 | Repartitioning interval |

The files each case reads and writes, and the record layout of each, are
described in `docs/BENCHMARKS.md`.
