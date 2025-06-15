# Comparison against TaylorDiff.jl

`taylordiff_bench.jl` times the directional task of the framework
comparison with [TaylorDiff.jl](https://github.com/JuliaDiff/TaylorDiff.jl),
the Julia implementation of Taylor-mode differentiation. On that task
it is the closest published relative of the method of this work:
forward Taylor propagation of the input derivatives, composed with one
reverse sweep (Zygote.jl) for the weight gradient. The JuliaDiff
organisation labels the package experimental.

The loss, network, weights and points are exactly those of the C++
directional benchmarks (`wx_ref.txt` is the same dump), so the printed
`computed loss` and `dL/dw_1` must match the reference table in
`bench/cpp/README.md`; the script checks this and refuses its timing
otherwise.

    julia> ] add TaylorDiff Zygote
    $ julia -O3 taylordiff_bench.jl <K> <repeat>

This benchmark is shipped ready to run but is **not part of the
measurements in the manuscript**: the measurement environment used
there could not install Julia. Numbers are only comparable within one
machine and one session, so to place TaylorDiff.jl into the table run,
back to back on the same machine:

    ./build/dir_grad_timing.out <K> <repeat>          # this work
    julia -O3 bench/julia/taylordiff_bench.jl <K> <repeat>

and, if built, the C++ directional benchmarks of `bench/cpp/`.
