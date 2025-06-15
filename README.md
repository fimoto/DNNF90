# DNNF90

Analytical propagation of high-order mixed partial derivatives through
feed-forward neural networks, with an exact adjoint for training on any
loss built from those derivatives.

A single forward pass carries every mixed derivative
`d^a N / dx^a` up to a prescribed multi-index set, and a single
analytical reverse pass returns the weight gradient of a loss defined on
them. The derivatives are exact to rounding, as with nested automatic
differentiation, but the cost is a fixed multiple of one ordinary
forward and backward pair and no nested computational graphs are built.

The recursions come from the multivariate Faa di Bruno formula organized
by multi-index partial Bell polynomials; `docs/` carries the derivations
and the full option semantics.

## Features

- Mixed partial derivatives to arbitrary order `K` in `D0` inputs,
  restricted automatically to the downward closure of the multi-indices
  a residual actually needs.
- Exact adjoint: the weight gradient of any differentiable functional of
  the carried derivatives, supplied through a seed interface.
- A physics-informed trainer for nonlinear PDE residuals, benchmarked on
  the Korteweg-de Vries and Zakharov-Kuznetsov hierarchies up to seventh
  order in four independent variables.
- Eleven gradient-based optimizers, including a damped natural-gradient
  method that solves the dense Gauss-Newton metric directly.
- An embeddable library layer with a C interface, a committee wrapper for
  uncertainty estimates, and a per-pattern extended Kalman filter.
- Built-in verification: finite-difference checks of the forward
  derivatives and of the analytical weight gradient, run on demand from
  the input file.
- Optional MPI build; optional BLAS acceleration.
- Fortran 2008: the code uses the `erf` intrinsic (in `lib/erf_value_module.f90`)
  and allocatable derived-type components, and nothing newer.  Everything
  except that module and the imported Bessel routines also passes a strict
  Fortran 2003 check (`make f2003check`).
  The only features used beyond Fortran 95 are allocatable components
  of derived types (TR 15581) and, in the C interface, `iso_c_binding`.

## Requirements

- A Fortran compiler (developed with gfortran 13; any F95 compiler works)
- Optional: an MPI implementation, a BLAS library, a C compiler for the
  C interface, Python 3 with PyTorch for the comparison scripts

## Build

```sh
make                 # serial trainer            -> build/serial.out
make BLAS=1          # serial trainer with BLAS
make mpi             # MPI trainer               -> build/mpi.out
make lib             # embeddable library        -> build/libdnnf90.a
make f2003check      # strict Fortran 2003 check (erf and Bessel modules excepted)
```

All build products go to `build/`; `make clean` removes it.

Example programs are separate targets, for instance
`make train_example.out`, `make customloss_example.out`,
`make libverify_example.out`, `make c_example.out`.

## Quick start

```sh
cd bench/kdv        # the third-order KdV benchmark, with an exact soliton
../../build/serial.out
```

The run writes every scalar training metric to a single
`history_ep0000000.dat` (self-describing header, one row per validation
event) and the model output to `output_set0001.dat`. Every benchmark directory contains
the input file, the data, and a reference log.

A minimal high-order fit with the self-check enabled:

```sh
cd zwork/hod_4d_k3
../../build/serial.out | grep PASSED
```

Every case directory under `zwork/` also ships a `job.sh` that runs it
as it stands (`cd zwork/pinn_zk7 && sh job.sh`), including the
multi-stage ones.

## Repository layout

```
lib/       embeddable library: Bell tables, evaluation, training, C API,
           committee and Kalman-filter wrappers
app/       keyword-driven trainer built on the library
host/      reference descriptors for the force-field example
tools/     example programs, benchmarks and data generators
utility/   weight-file conversion
bench/     benchmark suite with reference logs and reproduction scripts
zwork/     small demonstration cases, including keyword tours
docs/      user documentation
```

## Documentation

Everything in `docs/`:

- `docs/INPUT_KEYWORDS.md` - reference for every input keyword
- `docs/USAGE.md` - data formats, output files, restart, MPI, embedding,
  and the standalone measurement tools
- `docs/TUTORIAL.md` - setting up a differential equation of your own,
  from a blank directory to a verified solution, on the three-variable
  second-order case that ships as `zwork/tutorial_heat2d`
- `docs/BENCHMARKS.md` - the cases in `bench/` and `zwork/`: what each
  one shows, and the format of every input and output file it uses
- `docs/DESIGN.md` - loss interface, verification strategy, cost model
- `docs/VERIFICATION.md` - every self-check with the command that runs
  it and the number a clean build produces, the solved reference cases,
  and which optimizer to choose for which problem
- `docs/REPRODUCING.md` - how to recompute every figure and table of the
  accompanying paper; `tools/repro_check.py` verifies that the inputs
  for all of them are present in this tree
- `docs/SPECIFICATION.tex` - the specification: the derivations, the
  software structure, the library API, and the full option semantics
  (compiles with plain `pdflatex`)
- `zwork/FLOW_CASES.md` - the flow and coupled-system cases: what
  converges them, with the measured numbers

## Verification

The distribution verifies itself in five independent ways.

1. `Hod_check 1` in any input file compares the carried derivatives with
   finite differences of the network output, and the analytical weight
   gradient with finite differences of the full loss.
2. `make libverify_example.out` checks the library path against the
   trainer: one-point gradients against finite differences, threaded
   accumulation agreeing to rounding, and the Kalman filter reproducing
   the regularized normal-equation solution on a linear network to
   machine precision.
3. `make negtests` asserts the input-rejection contract, that each
   malformed or incomplete input stops with a diagnostic instead of
   proceeding, and properties that must hold exactly, for instance that
   a committee of identical members reports exactly zero spread, which
   the textbook variance formula does not.
4. `make harden` rebuilds everything with array-bounds checking, with
   uninitialized values poisoned by a signalling NaN, and with IEEE
   traps on invalid operations, division by zero and overflow.  Run the
   cases you care about against it.  This is the net for the defects a
   numerical comparison cannot expose: a write one element past an
   array, a read of a variable that was never set, a division that is
   only sometimes by zero.
5. Every benchmark case under `bench/` holds a reference log produced by
   the released code; `bench/speed_compare` is a timing protocol rather
   than a case, and carries its own README instead.

## Citation

If this code is useful in your work, please cite the accompanying
manuscript. The original code base was developed for the
doctoral thesis of F. Imoto, "Development of Orbital-Free Density
Functional Theory with Machine Learning and Its Applications", The
University of Tokyo (2019); the present version is a substantially
modified derivative of it.

## License

MIT License. Copyright (c) 2026 Fumihiro Imoto. See `LICENSE`, which also
records the exception: the Bessel routines in `lib/bessel_module.f90` derive
from Zhang and Jin's *Computation of Special Functions* and carry their own
acknowledgement rather than the MIT terms.
