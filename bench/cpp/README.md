# Comparison against C++ automatic differentiation

Two benchmarks against operator-overloading AD libraries in C++. They
time the same quantity as `bench/post/compare_frameworks.py`: **one
gradient of a loss with respect to every network weight**, where the
loss is built from derivatives of the network with respect to its
inputs. The network is 4-8-8-1 with tanh, twenty points, one core,
double precision.

That quantity, and not an epoch, is what the comparison is about. An
epoch of the Fortran trainer covers all training points in batches, so
comparing an epoch against a single gradient overstates the other side
by the number of batches.

## Neither library is distributed here

Both are obtained separately, and both are copyleft:

| library | licence | source |
|---|---|---|
| CoDiPack | GPL-3.0 | `github.com/SciCompKL/CoDiPack` |
| XAD | AGPL-3.0 (commercial licence also offered) | `github.com/auto-differentiation/xad` |
| ADOL-C | EPL-1.0 or GPL | `github.com/coin-or/ADOL-C` |

Only the benchmark sources are here, under the licence of this
distribution. They `#include` the libraries' headers; no code from
either library is copied into this tree, and building them is the
reader's choice.

## Building

CoDiPack is header-only. The order is fixed at compile time, one
binary per order, so that exactly one nesting depth is instantiated
and the optimizer sees the same small function a hand-written
single-order benchmark would give it:

    git clone --depth 1 https://github.com/SciCompKL/CoDiPack.git
    for K in 1 2 3 4 5 6 7; do
      g++ -O2 -std=c++17 -DORDER=$K -I CoDiPack/include \
          -o codi_bench$K codipack_bench.cpp
    done
    ./codi_bench3 <npoints> <repeat>

All seven orders run; K = 7 takes tens of seconds per gradient on one core.

XAD generates two headers with CMake, so it is configured first. The
library ships tape instantiations up to second order only, so the
benchmark compiles XAD's `Tape.cpp` inside `xad_tape_higher.cpp` and
adds the deeper stacks -- do **not** also link `-lxad`, or the standard
instantiations are duplicated:

    git clone --depth 1 https://github.com/auto-differentiation/xad.git
    cmake -S xad -B xad/bld -DCMAKE_BUILD_TYPE=Release -DXAD_ENABLE_TESTS=OFF
    make -C xad/bld
    g++ -O2 -std=c++17 -I xad/src -I xad/bld/src -o xad_bench \
        xad_bench.cpp xad_tape_higher.cpp
    ./xad_bench <order> <npoints> <repeat>

Orders 6 and 7 throw `std::bad_alloc` on a 4 GB machine: the payload
doubles per level and the tape no longer fits.

ADOL-C is built with CMake and needs C++20 for its headers:

    git clone --depth 1 https://github.com/coin-or/ADOL-C.git
    cmake -S ADOL-C -B ADOL-C/bld -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    make -C ADOL-C/bld
    g++ -O2 -std=c++20 -I ADOL-C/ADOL-C/include -I ADOL-C/bld/ADOL-C/include \
        -o adolc_tensor adolc_tensor_bench.cpp -L ADOL-C/bld -ladolc
    LD_LIBRARY_PATH=ADOL-C/bld ./adolc_tensor <K> <npoints> <repeat>

The Fortran side of the same measurement is `tools/fwd_grad_timing.f90`:

    ./build/fwd_grad_timing.out 4 <K> 8 20 100

## What the comparison can and cannot say

Both libraries reach high order by nesting their types, an adjoint tape
whose scalar is itself a forward type and so on. That is the same
construction as nesting first-order differentiation in a framework, so
the growth with the order is of the same kind; what a C++ tool changes
is the constant, since no interpreter is in the loop.

Two limits are worth stating plainly.

The nested C++ benchmarks take **one directional derivative**, not the
full set of mixed partials, so they are given the easier task. The
loss is the same for every implementation of the directional task,
including `tools/dir_grad_timing.f90` and the ADOL-C construction
below: the squared K-th directional derivative summed over the batch.

## Checking that a run is a real computation

Every benchmark prints its computed loss and one gradient component
(`dL/dw_1`) before the timing, so that a run is evidently the intended
computation: a benchmark that copies a not-yet-propagated adjoint
still times something, and prints a zero loss while doing it. The
shared `srand(12345)` weights make the
values reproducible; against an exact Taylor-jet evaluation of the
same network they are, to all printed digits:

| K | computed loss | dL/dw_1 |
|---|---|---|
| 1 | 7.539098907068e-05 | -3.00304789e-04 |
| 2 | 5.862671447015e-08 | -4.05874717e-07 |
| 3 | 7.890390176833e-08 | -5.18918334e-07 |
| 4 | 7.663173332728e-10 | -6.19269886e-09 |
| 5 | 1.158632876420e-09 | -7.04080029e-09 |
| 6 | 5.408322663523e-11 | -3.42215171e-10 |
| 7 | 8.175306274572e-11 | -3.10379965e-10 |

(The last digit of `dL/dw_1` may differ by one: the reference is a
central difference.) One asymmetry of intended use is kept: the ADOL-C
timing records its tape once and reuses it across repetitions, which
is that library's amortized mode, while CoDiPack and XAD record their
tape inside every repetition, which is theirs. Each library is used as
designed. The Fortran drivers use their own random weights,
so they print a different loss; their kernels are verified by the
package's own derivative checks (`make fdcheck`).

**ADOL-C needs a longer note.** Its `tensor_eval` builds the full
high-order derivative tensor by Taylor arithmetic rather than by
nesting, so it is the closest published relative of the method used
here, and `adolc_tensor_bench.cpp` measures it. On the forward pass it
is fast: all 330 mixed partials at K = 7 in 0.26 ms per point against
0.45 ms here.

But that driver answers a different question. `tensor_eval`
differentiates with respect to the **inputs** at a fixed set of weights,
and returns numbers. It does not record how those numbers depend on the
weights, so a loss built from them cannot be differentiated back to the
parameters, which is what training needs. There is no standard ADOL-C
driver that returns the weight gradient of a loss assembled from
high-order input derivatives; the two operations belong to different
parts of the library and are not composable.

`adolc_grad_bench.cpp` therefore does something else: it puts the
weights on the tape as independent variables and builds the input
derivatives inside it -- the exact Taylor recurrence of tanh, verified
against Taylor-mode differentiation to machine precision -- so that one
reverse sweep gives the weight gradient, which is checked against
central differences. That construction is written by hand and is close
to the recursions of this work expressed in ADOL-C's type, rather than
a use of ADOL-C as intended: read it as the cost of that construction,
not as a measurement of a library driver. It takes one directional
derivative rather than the full set; the Fortran side of that same
directional task is `tools/dir_grad_timing.f90`:

JAX's Taylor mode is timed on the same task and the same loss by

    python3 bench/post/compare_frameworks.py --setting dir --kmax K

which propagates the truncated jet along one direction with
`jax.experimental.jet` and differentiates the resulting loss.  It
completes K = 1 to 4 on the measurement machine and exhausts its
memory beyond that.


    make dir_grad_timing.out
    ./build/dir_grad_timing.out <K> <repeat>

A ready benchmark of the same task against TaylorDiff.jl, the Julia
Taylor-mode package, is in `bench/julia/`; it validates against the
reference table above.
