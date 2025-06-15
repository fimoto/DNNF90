# Paired activation study

Runs one case with several activations over several seeds and reports
the comparison. The four runs of a seed share the data, the network
shape, the schedule and the random draw, so the seed differences cancel
and the activation is what is left.

    export DNNF90_ROOT=/path/to/DNNF90     # optional; inferred otherwise
    ./run_study.sh 10 TANH SIN             # 20 runs, about a minute
    ./run_study.sh 5 TANH SIN BESSEL BESSEL1

The case shipped here is `zwork/hod_4d_k3`: four inputs, `K = 3`, so
thirty-five carried slots. It was set up for `tanh`, and its learning
rate and loss weights are unchanged, so any tuning in it favours `tanh`.

## What it reports, and why the worst slot matters

Three numbers per run: the coefficient of determination averaged over
the slots, the same for the least well fitted slot, and the final loss.

The average is the misleading one. A model can average 0.87 over
thirty-five slots while one of them carries no information, and it is
the worst slot that gives way first when an activation's high
derivatives grow. On this case, over ten seeds:

| activation | R2 mean (median) | R2 worst (median) | R2 worst (min) | loss (median) |
|---|---|---|---|---|
| `TANH` | 0.867 | 0.240 | **-0.761** | 6.3e-03 |
| `SIN` | 0.951 | 0.730 | 0.571 | 1.5e-03 |
| `BESSEL` | 0.989 | 0.860 | 0.539 | 3.7e-04 |
| `BESSEL1` | 0.983 | 0.858 | 0.527 | 1.3e-03 |

Each of the three bounded activations wins on all ten seeds, a sign test
at p = 0.001, and for each the worst-slot range is disjoint from the
`tanh` range. The `tanh` worst slot is negative on four seeds of the ten,
meaning the model does worse there than predicting the mean of the
target.

This is the third order. The same comparison at `K = 7`, where the
adjoint reaches order eight and the largest value of `|sigma^(8)|` is
1220 for tanh against 0.27 for `J_0`, separates much further; see
`zwork/hod_4d_k7_order7`.

## Reproducing

Two runs are kept here as a sample, `run_11111_TANH` and `run_11111_BESSEL`,
with the files summarise.py needs: the input, the history, the slot
predictions and the multi-index order. Running the script produces the
rest. `summarise.py` reads them and needs no rerun either.
