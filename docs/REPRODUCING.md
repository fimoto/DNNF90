# Recomputing the paper

Every figure and every table of the manuscript is produced from this
distribution. `tools/repro_check.py` verifies that the inputs are all
present; this file says how to run them.

    python3 tools/repro_check.py      # are the inputs there?
    make                              # build first; everything below needs it

Each case directory under `zwork/` ships a `job.sh` that runs that case
as it stands, so `cd zwork/<case> && sh job.sh` is the short form of
every `zwork` entry below; the multi-stage cases (`morse_refine`,
`tour_committee`, `act_seed_study`) have their whole sequence in it.

## Figures

| figure | how |
|---|---|
| solution, convergence, scaling, optimizers | `bench/run_all.sh` then `python3 bench/post/make_figs.py` from `bench/` |
| ten-dimensional case | run `bench/eyu10d`, then `python3 bench/post/make_fig_eyu10d.py` |
| force-field refinement | `cd zwork/morse_refine && sh job.sh` (see its README.md), then `python3 bench/post/make_fig_refine.py <root>` |

## Tables

| table | how |
|---|---|
| the benchmark suite | `bench/run_all.sh`; the summary it writes carries the columns |
| the PyTorch comparison | `python3 bench/post/torch_pinn.py` (needs torch; it says so if absent) |
| one weight gradient across frameworks and libraries | see below |
| activation derivative growth | `python3 tools/sigma_growth.py --latex` (needs mpmath) |
| teaching only the seventh order | run `zwork/hod_4d_k7_order7` over seeds, then `python3 bench/post/hod_accuracy.py <dir>` |
| third order, ten seeds | `cd zwork/act_seed_study && sh job.sh` (= `./run_study.sh 10 TANH SIN BESSEL BESSEL1`) |
| the non-sinusoidal collocation case | run `zwork/pinn_cosh7` over seeds; the solution R2 is the last two columns of `output_set0002.dat` |
| refinement against its control | `cd zwork/morse_refine && sh job.sh`, then `hod_accuracy.py` on each arm |


## The runs the text quotes outside the figures and tables

These are the settings the manuscript describes in prose. Each ships as
an input file next to the case it varies, with a header saying which
sentence it belongs to.

| text | how |
|---|---|
| Sec. 5.4, the floor against the batch size | `cd bench/kdv && sh run_scans.sh batch` |
| Sec. 5.4, the floor against the learning rate | `cd bench/kdv && sh run_scans.sh lr` |
| Sec. 5.4, the same ordering on 2-D Poisson | run `zwork/pinn_poisson2d` with `input_nn.dat` (Adam), `input_sgd.dat`, `input_ngd.dat`, and `zwork/pinn_poisson2d_kalman` with `input_nn.dat` |
| Sec. 5.4, the node-wise decoupled filter | `zwork/pinn_poisson2d_kalman/input_decoupled.dat` |
| Sec. 5.6, the residual without boundary data | `zwork/pinn_kovasznay/input_no_boundary.dat`, started from `fitted_weight.dat` |
| Sec. 5.7, eight curvature pairs against forty | `zwork/pinn_ehd/input_cold_m8.dat` against `input_cold.dat` |
| Sec. 5.7, Adam from the fitted state | `zwork/pinn_ehd/input_adam_2d-3.dat`, `input_adam_2d-4.dat` |
| Sec. 5.7, the natural gradient on batches of 40 | `zwork/pinn_ehd/input_cold_ngd_b40.dat` |
| Sec. 5.7, the epoch cost of the primal solve | `zwork/pinn_ehd/input_cold_ngd_primal.dat` against `input_cold_ngd.dat` |
| Sec. 5.7, the filter and its forgetting factor | `zwork/pinn_ehd/input_kalman_colloc.dat`, `input_kalman_colloc_lambda1.dat`, `input_kalman_colloc_lambda1_decoupled.dat` |
| Sec. 6, the hundredfold rate scan for tanh | `cd zwork/hod_4d_k7_order7 && sh run_lr_scan.sh` |
| Sec. 6, how far the seventh derivatives are | `make hod_dump.out`, then in `bench/zk7`: `../../build/hod_dump.out nn_weight.dat colloc.dat 4 7 5 > hod_dump.dat && python3 ../post/zk7_seventh.py hod_dump.dat` |
| Sec. 4.1, the forward-only cost per point | `make fwd_only_timing.out && ./build/fwd_only_timing.out 4 7 8 20 50` |
| Sec. 5.1, the exact coefficients | `python3 tools/check_exact_coeffs.py` |

The batch and rate scans write into `scan_*/` subdirectories, and the
rate scan of Section 6 into `lr_x*/`, so none of them disturbs the run
that ships in the case directory.

## The speed comparison table

Every column of that table must be measured **in one session, back to
back**, and reported as a median of repeated passes: the effective CPU
speed of a shared machine drifts by tens of per cent between sessions,
so numbers taken on different days are not comparable even on the same
host. With the C++ libraries built as `bench/cpp/README.md` describes:

    OMP_NUM_THREADS=1 python3 bench/post/compare_frameworks.py --setting hod --kmax K   # K = 1..4
    ./build/fwd_grad_timing.out 4 K 8 20 <repeat>     # this work, full mixed-partial set
    ./build/dir_grad_timing.out K <repeat>            # this work, directional
    ./codi_benchK 20 <repeat>                         # CoDiPack, one binary per order
    ./xad_bench K 20 <repeat>                         # XAD
    LD_LIBRARY_PATH=<adolc>/bld ./adolc_grad K 20 <repeat>   # ADOL-C

Every benchmark prints its computed loss and one gradient component
before its timing; the directional ones must match the reference table
in `bench/cpp/README.md` to all printed digits, and
`compare_frameworks.py` cross-checks PyTorch against JAX on a shared
batch. A run whose check fails is not a measurement. `bench/julia/`
adds TaylorDiff.jl to the directional group on the same terms; it is
not part of the published table.

## What is not regenerated here

The manuscript's `bench_table.tex` is a LaTeX fragment kept with the
manuscript sources rather than in this tree; the numbers in it come from
`bench/run_all.sh`.

Timings depend on the machine. The figures are drawn from whatever the
runs produce, so a rerun on different hardware changes the wall-clock
columns and nothing else.

## Seeds

Where the paper reports several seeds, the seed is set by `Rand_seed` in
the input file, and the values used are `11111 22222 33333 44444 55555`
and, for the ten-seed table, `66666 77777 88888 99999 12345` as well.
`zwork/act_seed_study/run_study.sh` walks that list.
