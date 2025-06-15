# Design notes

## Three layers

1. **Propagation.** The layer sweep that carries the derivatives is
   independent of the loss. It is driven by the Bell tables built once
   at start-up in `lib/multi_index_bell_module.f90`.
2. **Seed.** A loss enters only through dL/dT_a at the output layer, an
   `|A|`-dimensional derivative computed by the caller. This is the same
   separation as between an automatic-differentiation engine and a loss
   function in a machine-learning framework.
3. **Menu.** The trainer offers value fitting, derivative fitting and
   PDE residuals as keyword-selected presets on top of the seed layer.

The practical consequence is that value fitting, derivative fitting, PDE
residual training and Sobolev regularization are one loss with different
weights and targets, not four code paths.

## Where the library ends: mechanism, not policy

Minibatching, validation, early stopping and the optimizer menu live
in the trainer, not in the library, and the boundary is deliberate:
the library provides mechanisms (evaluate one point, form the weight
gradient of one point from a seed, apply one optimizer step to given
arrays), while the trainer owns policy (how large a batch is, when to
validate, when to stop, what to write to disk). The reason is the
embedding case. A host code, a density-functional or molecular-
dynamics program, has its own outer loop, its own data flow and its
own parallel decomposition; a library that imposes an epoch loop, a
validation split and checkpoint files cannot be embedded in it.
Mechanisms can also stay pure and reentrant, which is what allows one
work space per thread to be the only requirement for concurrent
calls, and they admit hard verification: a per-point gradient can be
checked against finite differences and against an independent
implementation to the last bit, while no comparable check exists for
a stopping rule. The same split explains the optimizer counts: the
library carries the steps a host can use point-by-point without a
training loop -- SGD, Adam, and the extended Kalman filter of
`kalman_module` (dense or node-wise decoupled, with its gate, process
noise and iterated variants) -- while the schedule-heavy research
menu, including the natural gradient that must accumulate metric rows
over a batch and the L-BFGS that owns a line search, stays with the
loop it is coupled to.

The consolidation is complete: the library is the only propagation
engine in the tree.  Every forward pass and every weight gradient of
the trainer, of the validation cost, of the prediction output, of the
self-check and of the C API goes through `net_forward_point`,
`net_backward_point` or `net_grad_batch`.  The trainer keeps its own
weight array, because the weights are policy as well: checkpoints,
restarts, MPI averaging and the optimizer states are written in terms
of it, and mirroring it into the library network once per batch
iteration keeps those file formats independent of the library's
internal layout.  The library holds the only set of propagation kernels; the trainer has
none of its own.  Three configurations are outside what they support: a
non-identity output activation, more
than one output, and the separate first-derivative channel, which is
the K = 1 case of high-order fitting and is now expressed that way.

Nothing is handed between the trainer and the library through shared
state.  A point's carried derivatives come back in the caller's buffer
and the loss seed goes in as an argument, so the seed builders, the
residual evaluator and the library are all functions of what they are
given.  The practical consequence is that the per-point step of the
trainer is already thread ready: it needs a work space and an
accumulator per thread, which the library provides, and buffers that
are private to the thread, which are now local variables rather than
module arrays.

The trainer owning the master weight array creates one invariant: the
library network must hold the weights as they are now.  It is not left
to discipline.  Every write to `weight` or `weight_best` increments a
generation counter, the synchronization records the generation and the
array it came from, and every evaluation entry checks both before it
computes.  The check is one integer comparison, so it stays on in
production builds; without it a missed synchronization returns
plausible numbers computed from the weights of the previous step,
which is the worst failure mode this layer can have.

Correctness is therefore established against the definition rather
than against a second implementation.  The weight gradient is checked
by central differences at three levels: `tools/dbg_backward_fd.f90` on
a synthetic dense network, the first test of `train_example` on a
trained network with an arbitrary seed, and the built-in `Hod_check`
inside a real run, which now verifies the engine that actually ships.
The harness also covers the batched value path, both against central
differences and against the per-point engine, and is run in the loop
build and in the `BLAS=1` build, since those compile different kernels.
The benchmark histories are frozen columns, so a change that alters a
trajectory is visible immediately.  They are frozen from the default
build; a `BLAS=1` build reorders the summations, and while four of the
five benchmarks still reproduce bitwise, a long run may separate from
the frozen column at equal quality once a last-digit difference
appears.

## The composite loss and its terms

The training objective is a weighted sum of terms, and the input
declares each term on one `Loss_term` line: a cloud of points, a form
(`DATA`, `HOD_DATA` or `COLLOCATION`), and a weight. One minibatch draws
from the pooled points of all terms, which is stochastic descent on
that sum. A multi-material force field (one supervised term per
material) and a physics-informed solver (one data term plus one
residual term) are the same structure with different terms, and the
data-to-residual balance of the latter is the ratio of two weights.

A `COLLOCATION` term is a collocation cloud: it parametrizes where the
PDE residual is evaluated and carries no target values, so supervised
concepts do not attach to it. Fitting a first derivative is not a
separate mechanism either: it is the K = 1 case of a `HOD_DATA` term.

The validation split holds out a random fraction of the pooled points,
so a residual term is validated too: its held-out points measure how
well the residual generalizes off the training collocation. A small
term can in principle lose all its training points to the draw; the
input is rejected in that case, and validation metrics of a term with
no held-out points are written as zero rather than NaN.

The keyword vocabulary reflects the general problem rather than the
first one this code was written for, which was learning kinetic-energy
functional
derivatives of several materials in a meta-GGA enhancement-factor
form: each set was one material, and the set index had to be carried
because the finite-difference gradients live in the oblique
coordinates of that material's lattice. Functional-derivative training
of that kind is out of scope for this trainer. It belongs in a
dedicated host code that computes its lattice-aware seeds and calls
the library directly through `net_grad_point`, which is what the seed
interface exists for. The term list is here because the
physics-informed composite loss needs it on its own: initial data,
boundary data and collocation clouds are separately weighted terms.
There is no set-scoped machinery for rescaling, batch quotas or
per-set output forms: an emphasis region is simply a separate weighted
term.

## Multi-index set

Carrying every multi-index up to order K is exponential in the number of
inputs. The implementation instead carries the downward closure of the
multi-indices that the residual seeds, which it derives from the PDE
specification. For the seventh-order benchmark this reduces the cost per
epoch by a factor of six relative to the dense set, with identical
training trajectories.

Multi-indices are kept in lexicographic order and located by binary
search, so the descriptor counts of a force field (50 to 100 inputs)
remain feasible.

## Verification strategy

Three checks, each able to fail independently:

- **Forward.** Carried derivatives against central differences of the
  network output.
- **Adjoint.** The analytical weight gradient against central
  differences of the full loss, sampled over weights, with guaranteed
  coverage of the bias of the first neuron in every layer.
- **Cross-implementation.** The library path against the trainer path on
  the same weights, point and seed; these agree bitwise.

The first two run from the input file with `Hod_check 1`, so they are
available in any user run and not only in the test suite.

## Cost model

One forward pass carries `|A|` derivative components through each layer,
and one reverse pass returns the weight gradient. The cost is therefore
a fixed multiple of an ordinary forward and backward pair, the multiple
being set by `|A|` and by the Bell table size, and it does not grow with
the depth of a nested differentiation graph.

Memory is dominated by the per-layer scratch of size
`(width, |A|, K+1)`. For wide networks with large multi-index sets this
is tens of megabytes per work space, and a committee allocates one per
member.

## Allocatable components (no pointers)

Derived types use allocatable components throughout; the library
contains no `pointer` attribute at all. Two things follow. First,
performance: the kernels read their work arrays through the derived
types inside the hot loops, and the no-alias, contiguous guarantee of
allocatable components is worth about 30 per cent of a whole gradient
at `-O3` against the same code with pointer components (measured;
pointers force the compiler to assume any two components may overlap).
Second, safety: allocatable components are released automatically when
an object goes out of scope or is passed as `intent(out)`, so a leak
requires effort. The `*_free` routines remain the explicit way to
release an object early, and assignment of a type with allocatable
components is a deep copy -- each committee member therefore owns its
own (small) table set; see the comment in `net_init`.
