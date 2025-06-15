# Usage

## Running

The trainer reads `input_nn.dat` from the current directory. Every case
directory under `zwork/` ships a `job.sh` that runs it as it stands, so
the shortest path is:

```sh
cd zwork/pinn_zk7
sh job.sh            # writes a.log; the case takes about 15 s
```

`job.sh` is one line for a single-stage case and carries the whole
sequence for the multi-stage ones (`morse_refine`, `tour_committee`,
`act_seed_study`), including the intermediate weight copies and the
`rm -f gd_*.dat` that a restart from earlier weights requires. Cases
with alternative input files list them in the header of their `job.sh`;
copy one over `input_nn.dat` to use it. Running the trainer directly
works just as well:

```sh
cd zwork/pinn_zk7
../../build/serial.out
```

MPI:

```sh
mpirun -np 4 ../../build/mpi.out
```

With one rank and a full batch the MPI build reproduces the serial
trajectory bit for bit.

Two parallelizations live behind that command, chosen by the method.

The gradient rules (`Simple`, `Adam` and the rest) train locally and
average the weights every `Average_cyc_mpi` epochs: each rank draws
`Num_batch` points from its own share, so the effective batch is
`nprocs` times the setting, and the run is not the serial one.

The whole-set methods -- `LBFGS`, `LM` and `Natural_grad` -- take one
step from one state, so they are split rather than averaged. The sweep
over the training set (L-BFGS and LM) or the batch (the natural
gradient) is divided among the ranks and summed back, `Num_batch` keeps
its meaning as the size of the one batch the ranks hold between them,
and no weight averaging is applied because every rank has taken the same
step. The result is the serial one up to the order of the additions:
on the shipped cases the objective agrees to twelve or thirteen digits
at two and four ranks. Reproducing a run bit for bit therefore requires
the same rank count.

## Data formats

All data files are plain text, whitespace separated, one record per
logical line. Records may wrap over several physical lines; the reader
counts values and stops with the record number if one is short.

- Value data (`Math`): `x(1:D0)  y`
- Derivative data: `x(1:D0)  dy/dx(1:D0)`
- High-order data (`Math_hod`): `x(1:D0)  y_a` for every carried
  multi-index, in the canonical order written to `hod_alpha_order.dat`
  at the start of a run
- Collocation data (`Pinn`): `x(1:D0)` and, with `Exact_solution 1`, the
  exact solution in the next column
- System data (a `System` block with `m` components and `r` residuals):
  `data.dat` is `x(1:D0)  y(1:m)`; `colloc.dat` is `x(1:D0)`, then with
  `Sys_src` one source value per residual, then with `Exact_solution 1`
  the exact solution of every component. The reader counts values, so a
  wrong `Sys_src` setting stops with a record-length error rather than
  shifting the columns silently

## Output files

The number in the history file name is the epoch at which that run
started, so a restarted run opens a fresh file instead of overwriting
the previous one. Checkpoints carry the epoch at which they were
written.

| File | Contents |
|---|---|
| `history_ep0000000.dat` | All scalar training metrics, one row per validation event. The header names every column: epoch, patience, then cost, RMSE and MAE for the training, validation and full input sets, split into the value and derivative parts. With several training sets, per-set blocks of the same metrics follow. Columns 3-5 are the composite objective with the `Loss_term` weights applied — the same sum the optimizer descends, which is also what the patience rule, the `Conv` threshold and the best-weight selection read; the per-part and RMSE/MAE columns stay unweighted diagnostics |
| `output_set0001.dat` | Model output at the points of set 1 |
| `output_deriv_set0001.dat` | First derivatives, with `Output_deriv 1` |
| `output_hod_set0001.dat` | All carried derivatives, high-order runs |
| `checkpoint_ep0001000.dat` | Weights at epoch 1000, written every `Checkpoint_cyc` epochs |
| `nn_weight.dat` | Final weights, read by `Restart 1` |
| `gd_*.dat` | Optimizer state, read by `Restart 1` |
| `lbfgs_history.dat` | With `GD_method LBFGS`: epoch, the loss the line search accepted, the step length, the number of line-search trials, and whether a step was accepted. This is the objective the search itself minimizes. The history's cost columns carry the same Loss_term-weighted objective, so for a full-batch run the two agree per point to all printed digits; this file remains the per-iteration view (step lengths, trial counts, acceptances) |
| `hod_alpha_order.dat` | Canonical multi-index order of the run |

Every case directory under `zwork/` carries a `plot.gp`;
`gnuplot plot.gp` after a run writes `plot.png` with the case's fields
and its learning curve. The column layout of each output file is
repeated in a comment at the top of the script.

A checkpoint has the same format as `nn_weight.dat`, so copying one to
`nn_weight.dat` restarts from that epoch. A typical run therefore
leaves one history file, one output file per set, the checkpoints, and
the restart pair.

## Prediction without training

`Task PREDICT` loads `nn_weight.dat`, writes the output files for every
loss term (and the derivatives with `Output_deriv 1`), and exits. No
history or checkpoints are written.

## Verifying a build

    make fdcheck.out && ./build/fdcheck.out

checks the weight gradient of both propagation paths against central
differences, the batched path against the per-point one, and the
evaluation entry against the training forward.  Run it again with
`BLAS=1`, which compiles different kernels.

The in-run self-check, enabled with `Hod_check 1`, tags its lines by
what each test differentiates rather than by the method, since three of
the four are finite-difference comparisons:

| tag | what it compares |
|---|---|
| `[REF]` | every carried derivative against reference values from an independent implementation |
| `[dX]` | the low-order input derivatives against central differences of the network output, relative to their own size |
| `[dW]` | the weight gradient against central differences of the supervised-derivative loss |
| `[dW-R]` | the same against central differences of the collocation residual loss |

A failure stops the run. `bench/post/make_figs.py` reads the `[dW-R]`
value out of a run log, and accepts the older `[P]` spelling so that
archived logs still parse.

Two further checks cover what a gradient comparison cannot see.

    make harden      # bounds checking, poisoned uninitialized values,
                     # IEEE traps; then run the cases you care about
    make negtests    # input-rejection contract and exact properties

`negtests` asserts that each malformed or incomplete input stops with a
diagnostic instead of proceeding, and that properties which must hold
exactly do: a committee of identical members reports exactly zero
spread, for instance, which the textbook variance formula does not.

The two builds compute the same quantities in a different summation
order.  Gradients agree to roundoff, and most benchmarks reproduce
bitwise, but a long training run can separate: `bench/kdv` is bitwise
identical for sixty validation cycles and then diverges at equal
quality after a last-digit event.  Numbers quoted from a run should
therefore name the build they came from.  Inside a real
run, `Hod_check 1` performs the same kind of check on the actual data
and loss.

## Measurement tools

Three standalone tools ship in `tools/` beside the verification pair:

    make fwd_grad_timing.out    # phase-resolved cost of one gradient
    ./build/fwd_grad_timing.out D0 K width npoints
                                # contraction vs Bell-composition split

    make bell_pad_timing.out    # list vs padded composition kernels,
    ./build/bell_pad_timing.out D0 K width nrep
                                # bitwise-verified against the list
                                # kernel; padding measures 2.8-4.1x
                                # slower, so the list form is the
                                # reference kernel on CPUs

    python3 tools/ipinner_enkf.py --help
                                # the observation-space EnKF analysis of
                                # an iPINNER-style outer loop for noisy
                                # data with imperfect physics; workflow
                                # and measured results in
                                # zwork/FLOW_CASES.md

## Restart

Set `Restart 1` and keep `nn_weight.dat` in the directory. If the
optimizer state files are absent the run starts from the stored weights
with a fresh optimizer, which is the intended behavior when changing
method. Learning-rate schedules and the Adam bias correction continue on
the absolute epoch counter, so a run split into two halves matches an
uninterrupted run of the same length.

## Embedding in a host code

`app/api_module.f90` exposes three calls: initialize from an input file,
evaluate the network and its carried derivatives at a point, and
finalize. The library layer in `lib/` is the more flexible entry point:
it holds no global state, so several networks can be trained and
evaluated in one process, which is what a force field with one network
per element needs.

The C interface is declared in `lib/dnnf90.h`; `tools/example_c.c` is a
complete example, and `tools/example_embed.f90` is the Fortran
equivalent.

## Custom losses

The library contract is loss agnostic. `net_grad_point(nt, tw, x, seed,
g)` returns the weight gradient for any differentiable functional of the
carried derivatives, given its seed dL/dT_a. The three forms of the
trainer (`Math`, `Math_hod`, `Pinn`) are presets on top of that
interface, not a limit on it. `tools/example_customloss.f90` trains a
Huber value loss combined with a Sobolev penalty, which the presets
cannot express, by changing three lines of seed code.

Note that `net_init` does not initialize the weights: that is the
caller's responsibility, and leaving them at zero makes the network
degenerate. The example contains a deterministic initialization to copy.

## Optional BLAS

`make BLAS=1` routes the layer contractions through `dgemm`. Results are
not bit-identical to the loop path, because the library chooses its own
summation order; both paths pass the finite-difference checks.
