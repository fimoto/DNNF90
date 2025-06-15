# Inputs and outputs of `bench/` and `zwork/`

A reference for the two trees of ready-to-run cases: every file a case
reads, every file it writes, the record layout of each, and what each
case is for. `docs/INPUT_KEYWORDS.md` is the keyword reference; this
document is about the files.

Every case is self-contained. Change into the directory and run the
binary; nothing is written outside it.

    cd zwork/x2+1
    ../../build/serial.out

---

## 1. The two trees

**`bench/`** holds the numerical experiments of the method paper. Five of
them (`kdv`, `zk7`, `g7`, `opt_ngd`, `opt_kalman`) also serve as the
regression net: column 3 of their history is compared bitwise against
frozen reference columns after any change to the propagation path. Their
inputs should not be edited casually.

**`zwork/`** holds demonstrations. Each exercises a feature of the input
language on a small problem that finishes in seconds, so it is the place
to look when writing a new input file.

Two `bench/` cases are instrumentation rather than experiments and their
output must not be read as a result; see section 6.

---

## 2. Input files

### 2.1 `input_nn.dat`

One key per line, `KEY value ... /`, with everything after `/` ignored
and `#` or `!` starting a comment line. Keys may appear in any order,
except that `Loss_term`, `Residual` and `Time_axis` need `Nlayer` to have
been read first. A repeated key takes the value of its last occurrence.
An unrecognized key produces

    input WARNING: unrecognized keyword ignored: <key>

and is skipped; a retired key stops the run with its migration path.

Only five keys are required: `Nlayer`, `Init_w`, `Loss_term`,
`GD_method`, `GD_param`. High-order and collocation runs add `Hod_K` and
`Residual`.

### 2.2 Data files named by `Loss_term`

`Loss_term FORM n file w /` reads `n` records from `file`. Records are
read with list-directed input, so **a record must be numbers only:
comment lines are not skipped**. Lines after record `n` are never read,
which is how the trailing `#Nt=` marker in `zwork/x2+1/train.dat`
survives.

`D0` is the number of inputs, that is the width of layer 1. `NUM_alpha`
is the number of carried multi-indices, listed in `hod_alpha_order.dat`.

| `FORM` | Record layout |
|---|---|
| `DATA` | `x(1:D0)`, `y` |
| `HOD_DATA` | `x(1:D0)`, `y_alpha(1:NUM_alpha)` in multi-index order, the first being the value |
| `COLLOCATION` | `x(1:D0)`, then the source `f(x)` if the residual declares a `SRC` term, then the exact solution if `Exact_solution 1`, in that order |

A `COLLOCATION` record carries no target: the point only says where the
residual is imposed. The exact-solution column, when present, is reported
on but never trained against.

Shipped names are `train.dat` for a value fit, `data.dat` for the
supervised part of a collocation case and `colloc.dat` for its interior
points, but the names come from the `Loss_term` line and nothing depends
on them.

Preparing a `HOD_DATA` file is circular at first sight: the file needs
the column count and the column order, and both come from the run that is
supposed to read it. `tools/alpha_order.py` breaks the circle by
reproducing the enumeration:

    tools/alpha_order.py --d0 4 --k 3                       # dense set
    tools/alpha_order.py --d0 4 --seeds path/to/seeds.dat   # closure
    tools/alpha_order.py --d0 10 --k 3 --count              # 286

Its output is byte-identical to the `hod_alpha_order.dat` the run writes,
for the dense set, for a seed closure and for the closure a residual
induces.

### 2.3 Auxiliary inputs

| File | Read when | Content |
|---|---|---|
| `alpha_seeds.dat` | `Hod_alpha_file` names it | first line the number of seeds, then one multi-index `alpha(1:D0)` per line. The carried set is their downward closure |
| `nn_weight.dat` | `Restart 1`, or `Task PREDICT` | weights, in the format of section 4.3 |
| `gd_dw.dat` and friends | `Restart 1`, if present | optimizer state; absent, the optimizer starts cold |
| member weight files | `Task COMMITTEE` | one per line after `Committee m /`, each in the format of section 4.3 |

`zwork/hod_4d_k7_active` is the case that uses `alpha_seeds.dat`: its
high-order fit carries the closure of a few seed indices instead of the
dense set.

`zwork/tour_committee` is the case with member files. It ships
`member_template.dat` and `train_members.sh`, which runs the template
four times with different `Rand_seed` and keeps each `nn_weight.dat` as
`member_<seed>.dat`, plus `committee.dat`, the input that evaluates them.
Copy `committee.dat` to `input_nn.dat` to run it.

---

## 3. Reading the history

`history_ep<start epoch, 7 digits>.dat`, one row per validation event,
not per epoch: a run with `Validation_cyc 100` and `Epoch 20000` produces
about two hundred rows. The file name carries the epoch the run started
from, so a restart writes a new file rather than appending. The `#` header
is self-describing; all metrics are per point.

Each metric has two channels. The **value channel** (`_f` in the header)
is the contribution of `N(x) - y`; the **derivative channel** (`_df`) is
that of the high-order targets and of the collocation residual. The
totals in columns 3 to 5 are assembled per form: `DATA` uses the value
channel only, `COLLOCATION` the derivative channel only, `HOD_DATA`
lambda_0 times the value plus the derivative part.

| Column | Content |
|---|---|
| 1 | epoch |
| 2 | patience counter, compared against `P_max` |
| 3-5 | total cost: training, validation, all input points |
| 6-9 | cost split: training (value, derivative), validation (value, derivative) |
| 10-11 | the same split over all input points |
| 12-17 | RMSE: training (value, derivative), validation, all input |
| 18-23 | MAE, in the same order |
| 24 and on | only when the loss has two or more terms: 21 columns per term, in the order of columns 3 to 23. Columns 24-44 are term 1, 45-65 term 2, and so on |

A `DATA` plus `COLLOCATION` run therefore has 65 columns, with the
derivative-channel columns of term 1 zero and the value-channel columns
of term 2 zero.

Useful one-liners:

    # training cost against epoch
    awk '!/^#/{print $1, $3}' history_ep0000000.dat

    # did the patience counter stop the run?
    grep -v '^#' history_ep0000000.dat | tail -1 | awk '{print "patience", $2}'

---

## 4. Reading the other outputs

### 4.1 Predictions

| File | Written when | One line holds |
|---|---|---|
| `output_set<nnnn>.dat` | end of run | `x(1:D0)`, prediction, target. One file per loss term, numbered in the order the `Loss_term` lines appear. For a `COLLOCATION` term the target column holds the exact solution when `Exact_solution 1` |
| `output_hod_set<nnnn>.dat` | `HOD_DATA` terms | `x(1:D0)`, all predicted derivatives `T^alpha(1:NUM_alpha)`, all target derivatives. Column order is that of `hod_alpha_order.dat` |
| `output_deriv_set<nnnn>.dat` | `Output_deriv 1` | `x(1:D0)`, `dN/dx_i` for `i = 1..D0` |
| `output_committee_set<nnnn>.dat` | `Task COMMITTEE` | `x(1:D0)`, the ensemble mean of every carried slot, then the sample standard deviation of every carried slot |

Because the last two columns of `output_set` are the prediction and the
target, the quality of a fit is one line of arithmetic:

    awk '!/^#/{d=$(NF-1)-$NF; s+=d*d; m+=$NF; n++; y[n]=$NF}
         END{m/=n; for(i=1;i<=n;i++) v+=(y[i]-m)^2; printf "R2 = %.6f\n", 1-s/v}' \
        output_set0001.dat

A value near 1 means the model explains the data; near 0 means it has
learned no more than the mean of the target. **This is worth running
after any change**: a network stuck at the mean still writes a smooth,
plausible looking output file. A negative value means the fit is worse
than the mean, which for the measurement cases of section 6 is expected
and for anything else is a problem.

For a `COLLOCATION` term whose target column is all zeros the formula
divides by zero; use the file of a case with `Exact_solution 1`, where
that column is the exact solution.

### 4.2 Reference information

| File | Content |
|---|---|
| `hod_alpha_order.dat` | one line per carried multi-index: slot number, order of the multi-index, then `alpha(1:D0)`. This defines the column order of every derivative file and the slot numbers in the residual echo. `tools/alpha_order.py` reproduces it without running a case, so a `HOD_DATA` file can be prepared before the first run |
| `data_division.log` | the indices of the points held back for validation |
| `hod_golden.dat` | reference values for the self-check; absent, the regression part of the check is skipped |

### 4.3 Weights and restart

| File | Content |
|---|---|
| `nn_weight.dat` | the best weights, measured on the training cost. Line 1 the best epoch, lines 2 and 3 reserved fields written as `func 0` and `Activation_out 0`, then the layer count, then one line per layer width, then for each layer `l >= 2` a `#l=` line followed by one line per neuron holding `W(l,j,0:n_{l-1})`, index 0 being the bias |
| `checkpoint_ep<nnnnnnn>.dat` | the same format, written every `Checkpoint_cyc` epochs, with line 1 the epoch of that checkpoint. Copy it over `nn_weight.dat` to restart from there |
| `gd_dw.dat`, `gd_m.dat`, `gd_v.dat`, `gd_r.dat`, `gd_u.dat` | the optimizer state: the last step and the moments each rule keeps. Which files appear depends on the rule |

`Restart 1` reads `nn_weight.dat`. If `gd_dw.dat` is present the
optimizer state is restored as well, so Adam resumes with its moments
rather than cold; if the epoch in the weight file and the epoch in the
optimizer log disagree, the run stops rather than mixing a weight file
with a mismatched optimizer state.

### 4.4 What a collocation run prints first

A run with a `Residual` block echoes the operator as the parser
understood it, before training starts:

    ### Residual as parsed (imposed as R = 0 at every collocation point)
    ###   axes are written x1..x2, with axis 2 shown as t (Time_axis)
    ###   term      coefficient  expression                    slot  multi-index
    ###    1      1.000000E+00  u_t                               3    0  1
    ###    2      3.000000E+00  u * u_x1                          2    1  0
    ###    3      1.000000E+00  u_x1x1x1                          5    3  0

The operator is the one quantity in the input that cannot be checked
against anything else: a mistyped multi-index yields a different
equation, solved correctly. Comparing this echo with the equation you
meant is the only guard. The slot column cross-references
`hod_alpha_order.dat`.

---

## 5. The cases

Timings are single core in the default build and are indicative only.

### 5.1 `zwork/`, demonstrations

| Case | Demonstrates | Problem | Time | Result |
|---|---|---|---|---|
| `x2+1` | the smallest input: one `DATA` term | fit `y = x^2+1` | 1 s | R2 = 0.999 |
| `tutorial_heat2d` | the worked case of `docs/TUTORIAL.md`: three variables, a closure instead of the dense set, boundary and initial data as a `DATA` term | 2-D heat equation | 30 s | R2 = 0.9965 against the analytic solution |
| `tour_derivfit` | fitting a first derivative as the `K=1` case of a high-order term | `y = x^2+1` with `dy/dx` targets | 2 s | R2 = 1.000 |
| `tour_sobolev` | Sobolev smoothing: a zero target on the third derivative | noisy sparse data | 1 s | R2 = 0.999 |
| `tour_schedule` | a decaying learning rate and the patience stop | small regression | 1 s | stops early by design |
| `act_seed_study` | a paired activation comparison over seeds, with a runner and a summary script; two sample runs are kept so the table can be checked without rerunning | `hod_4d_k3` | 60 s | at K=3 the periodic activation wins 10/10 and the worst slot is negative for tanh on four seeds |
| `morse_refine` | refining a force field trained on forces alone with high-order derivatives, with a control arm that adds the same epochs without them | Morse chain, six atoms | 3 x 60 s | cubic error 0.44 before, 0.02 after, 0.44 in the control |
| `tour_committee` | `Task COMMITTEE`: spread over four independently trained members | `y = x^2+1` | 40 s | spread 1.1e-2 (value), 2.3e-2 (derivative) |
| `hod_4d_k3` | high-order fitting, dense set, `D0=4`, `K=3` (35 targets per point) | product of sines | 4 s | R2 = 0.993 |
| `hod_4d_k7` | the same at `K=7` (330 targets per point) | product of sines | 55 s | R2 = 0.993 |
| `hod_4d_k7_order7` | teaching one derivative order and nothing else: lambda_p = 0 except lambda_7 = 1, which isolates what an activation can represent at high order | product of sines | 50 s | loss 0.24 with tanh against 0.04 with J_0; see the activation section of the paper |
| `hod_4d_k7_active` | a seed closure instead of the dense set | product of sines | 2 s | R2 = 0.981 |
| `pinn_kdv` | collocation on a one-dimensional evolution equation | KdV soliton | 8 s | R2 = 1.000 |
| `pinn_kdv_ngd` | the same with the natural-gradient rule | KdV soliton | 42 s | R2 = 1.000 |
| `pinn_zk7` | a seventh-order operator in four variables, the deepest chain shipped | ZK hierarchy | 29 s | R2 = 0.999 |
| `pinn_kovasznay` | the Kovasznay flow, an exact steady solution of the 2-D Navier-Stokes equations; three components, three residuals, twelve terms | Re = 40 | 3 min | R2 > 0.997 for u, v and p from the exact fit |
| `pinn_cavity` | the lid-driven cavity at Re = 100, compared with Ghia et al. (1982) via `post/cavity_compare.py`; the pressure is unconstrained at the wall and is given zero weight in the supervised term | Re = 100 | 25 min | centreline profiles of the right shape, rms 0.12 against the reference |
| `pinn_lax` | Lax's fifth-order equation, a scalar case that needs the `System` products: `TRP` for 30 u^2 u_x and `DXD` for 20 u_x u_xx. Exact soliton (verified in 40-digit arithmetic); term table read back independently by `bench/post/check_lax_input.py` (8e-16); `gen_data.py` regenerates the data. Cold full-batch L-BFGS: relative L2 = 1.0e-3, R2 = 0.999998 in ~106 s | exact | 2 min | the product language on a classical scalar equation |
| `pinn_lax7` | Lax's SEVENTH-order equation, the case Chen et al. (Sci. Rep. 14 (2024) 23874, Eqs. (20) and (53)) solve with nested differentiation. Its quartic term `140 u^3 u_x` is a product of four factors and needs `QAD`. Exact solitary wave `u = sech^2((x-t)/2)/2`; term table read back independently by `bench/post/check_lax_input.py --order 7` (1.1e-14); `Hod_check 1` verifies the four-factor adjoint against central differences (2.7e-10). Cold full-batch L-BFGS, 3000 epochs (about 40 s): relative L2 = 8.8e-4, R2 = 0.999999 | exact | 1 min | the four-factor term; a published point of comparison |
| `pinn_ehd`, `pinn_ehd_mild` | an electrohydrodynamic system: five components, five residuals, twenty-five terms, coupled both ways through the body force and the charge transport. `pinn_ehd` ships four inputs: `input_nn.dat` (the Kalman divergence case, kept as the system's reference term table -- read its header before running it), `input_cold.dat` (full-batch L-BFGS from random weights: 0.974 to 1.2e-5 per point in 6000 epochs, R2 >= 0.99995 all components), `input_cold_ngd.dat` (dual natural gradient on 120-point batches, per-residual rows: 0.970 to 1.1e-6 per point in 520 epochs), and `input_fit.dat` (supervised fit to the manufactured solution, producing the shipped `fitted_weight.dat`; `fit_data.dat` is columns 1,2,8-12 of `colloc.dat`) | manufactured | 5-15 min | term table verified to 1e-14; two independent optimizers solve it cold |
| `pinn_taylorgreen` | a coupled system: three field components over three variables, with the exact Taylor-Green solution. The momentum equations carry u u_x + v u_y, one component multiplying the derivative of another, which needs the System block | 2-D Navier-Stokes | 100 s | R2 = 0.95 for all three components at the collocation points |
| `pinn_disp7` | a seventh-order dispersive equation whose seventh-order term dominates, unlike the benchmark solitons where it enters weighted by 1.7e-6; exact solution a sine | u_t + u_7x = 0 | 65 s | R2 = 0.997 with SIN against 0.24 with tanh |
| `pinn_cosh7` | the same equation with a deliberately non-sinusoidal exact solution, which removes the advantage a periodic activation has when the answer is itself a sine | u_t + u_7x = 0 | 65 s | R2 = 0.956 with J_0 against 0.686 with tanh, five seeds |
| `pinn_poisson2d` | an inhomogeneous operator: a `SRC` term reads the source per point | 2-D Poisson | 60 s | R2 = 0.9999 against the analytic solution |
| `pinn_poisson2d_kalman` | the same problem trained by the Kalman filter | 2-D Poisson | 8 s | R2 = 1.0000, rms 1.8e-5 in 300 epochs |

`tour_schedule` is the one case that ends with a mediocre fit on purpose:
it exists to show `SIMPLE_SCHEDULE` and `P_max` taking effect, and it
stops when the patience counter runs out.

### 5.2 `bench/`, experiments

| Case | Problem | Time | Result |
|---|---|---|---|
| `kdv` | KdV soliton, `sech^2` | 4 s | R2 = 1.000 |
| `kawahara` | Kawahara equation, `sech^4` | 11 s | R2 = 1.000 / 0.999 |
| `g7` | seventh-order one-dimensional equation, `sech^6` | 16 s | R2 = 1.000 / 0.988 |
| `zk3` | Zakharov-Kuznetsov in three space dimensions | 1 s | R2 = 1.000 |
| `zk5` | fifth-order member of the same hierarchy | 4 s | R2 = 0.999 |
| `zk7` | seventh-order member | 18 s | R2 = 0.999 |
| `eyu10d` | ten-dimensional Laplace problem | long | R2 = 0.999 |
| `slit` | Laplace on a slit square, square-root singularity at the corner | 15 s | R2 = 0.81; the singularity limits the strong form, and the case ships as a documented limitation |
| `opt_simple` | the KdV problem with plain descent | 6 s | R2 = 0.967 / 0.815 |
| `opt_ngd` | the same with the natural gradient | 36 s | R2 = 1.0000 / 0.9998 |
| `opt_kalman` | the same with the extended Kalman filter | 6 s | R2 = 1.0000 / 1.0000 |

`opt_simple`, `opt_ngd` and `opt_kalman` are one experiment in three
parts: identical problem, data, network and seed, with only the rule and
the epoch budget changed.

| Rule | Epochs | Time | Training cost | R2 (data / collocation) |
|---|---|---|---|---|
| `SIMPLE` | 20000 | 6 s | 2.4e-3 | 0.9667 / 0.8154 |
| `NATURAL_GRAD` | 3000 | 36 s | 1.5e-5 | 0.999993 / 0.999803 |
| `KALMAN` | 1000 | 6 s | 2.6e-13 | 1.0000000 / 1.0000000 |

---

## 6. The two measurement cases

`scal_closure` and `scal_dense` compare the cost of a seed closure
against the dense multi-index set on the same problem. Validation is
switched off with `Validation_cyc 1000000`, the run stops after 500
epochs, and **the fit is meaningless by construction**: R2 comes out
around -15. What these cases measure is the time and the multi-index
count reported on startup, not the fit.

`speed_compare` measures plain training throughput at width 768 against
PyTorch and neural-fortran. `Epoch 5` and `job.sh` drive it;
`torch_mirror.py` and `nf_bench.f90` are the other two implementations of
the same five steps. `input_small.dat` is the narrow-width variant. Its
output is a timing, not a fit.

---

## 7. Checking a run

Three things are worth looking at in any log.

1. `ALL PASSED` from the in-run self-check, in the cases that set
   `Hod_check 1`. It compares the gradient the run is about to use
   against central differences of the loss, so it verifies the engine
   that actually trains.
2. No message about synchronization. That guard fires when the library
   network holds weights older than the trainer's, and stops the run
   rather than reporting numbers computed from stale weights.
3. The R2 of `output_set*.dat`, by the recipe in section 4.1.

A quick pass over a whole tree:

    for d in zwork/*/; do
      ( cd $d && ../../build/serial.out > run.log 2>&1
        echo "$d rc=$? $(grep -c 'ALL PASSED' run.log)" )
    done

Reproducing the frozen reference columns of the five regression cases
requires the default build: a `BLAS=1` build reorders the summations, and
while gradients agree to roundoff and most benchmarks still reproduce
bitwise, a long run can separate from the frozen column at equal quality
once a last-digit difference appears. Numbers quoted from a run should
name the build they came from. See `docs/USAGE.md`.

---

## 8. The figures of the paper

`bench/post/` holds the scripts that draw them and the numbers they
report.

    cd bench
    python3 post/make_figs.py         # solution, convergence, optimizers, scaling
    python3 post/make_fig_eyu10d.py   # the ten-dimensional case

`post/make_fig_taylorgreen.py <run dir>` draws the coupled-system case: the exact Taylor-Green vortex, the network, and their difference on the same colour scale, one column per field component.

`post/make_fig_refine.py <root>` draws the refinement result: the relative error of each derivative order for the forces-only fit, the control and the refined model, with the line at one that marks where a model stops carrying information about an order.

`tools/example_ngd_multi.f90` checks the natural-gradient metric row of a coupled system against differences of the loss, and `tools/example_kalman_multi.f90` checks that one filter update shrinks the residual it was given. Both are the checks that caught sign and slice errors in the multi-component paths.

`post/check_kovasznay.py <case>` and `post/check_system_input.py <case>` read a System block back and evaluate it at the exact solution, which is how a mis-transcribed term is caught before any training. `post/cavity_compare.py` plots the cavity centrelines against Ghia et al.

`post/compare_frameworks.py` times one weight gradient of a loss built from input derivatives, in PyTorch and in JAX, both nested and with Taylor mode; it cross-checks the two against each other on a shared batch before reporting any timing. `bench/cpp/` does the same against CoDiPack, XAD and ADOL-C on the directional task; none is distributed here and `bench/cpp/README.md` says how to obtain them, how to build each benchmark (CoDiPack needs one binary per order, `-DORDER=K`; XAD needs the extra translation unit `xad_tape_higher.cpp` and no `-lxad`), and what loss and gradient value a correct run must print. `bench/julia/` is the same directional task against TaylorDiff.jl. `tools/fwd_grad_timing.f90` is the Fortran side of the full mixed-partial measurement and `tools/dir_grad_timing.f90` of the directional one.

`post/hod_accuracy.py <run dir>` reports the accuracy of a high-order
fit resolved by derivative order, which a single coefficient of
determination over all carried slots cannot show: on the K = 7 case the
targets are the same size at every order while the predictions diverge at
the top, so one number hides the structure.  It reports, per order, the
relative error and the amplitude ratio against the targets, and flags the
orders that are lost or diverging.  It reads the run products and needs
no rerun.

`post/make_fig_forcefield.py` draws the force-field demonstration from the
log of `hod_ff_example.out`.  That demonstration is not part of the paper;
it ships as a worked example of training on second derivatives, and its
figure is produced only if you ask for it.

Both read only run products: `a.log` (for the multi-index count, the [dW-R]
self-check value and the wall time), `history_ep0000000.dat`,
`nn_weight.dat` and `colloc.dat`, per case. Nothing is retrained. The
cases they need are `kdv kawahara g7 zk3 zk5 zk7 opt_simple opt_ngd
opt_kalman scal_closure scal_dense slit eyu10d`, so those must have been
run with their standard output kept as `a.log`:

    ( cd bench/kdv && ../../build/serial.out > a.log 2>&1 )

`figs/summary.txt` collects every number quoted in the paper and
`figs/bench_table.tex` is the summary table it includes.
