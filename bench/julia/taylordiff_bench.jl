# This file (bench/julia/taylordiff_bench.jl) is part of DNNF90.
# (MIT License; see LICENSE at the repository root.)
#
# Weight gradient of the directional-derivative loss with TaylorDiff.jl,
# the Julia implementation of Taylor-mode differentiation.  On the
# directional task it is the closest published relative of the method of
# this work: forward Taylor propagation for the input derivatives,
# composed with one reverse sweep (Zygote) for the weight gradient.
#
# The loss is  sum_n (d^K u/dx_1^K)^2  over the batch -- the same loss
# every column of the directional group of the comparison table
# differentiates.  Weights and points are read from wx_ref.txt, the
# shared dump used by the C++ benchmarks, so the computed loss and
# dL/dw_1 must match the reference table in bench/cpp/README.md to all
# printed digits; the script checks this before timing anything.
#
#   julia> ] add TaylorDiff Zygote
#   $ julia --project -O3 taylordiff_bench.jl <K> <repeat>
#
# For a fair comparison run tools/dir_grad_timing.out (and the C++
# benchmarks) on the same machine in the same session: the table in the
# manuscript was measured that way, and machine-to-machine or
# session-to-session numbers are not comparable.

using TaylorDiff
using Zygote
using Statistics
using Printf

# ---- shared weights and points -----------------------------------------
function read_wx(path)
    toks = split(read(path, String))
    npt = parse(Int, toks[2]); nw = parse(Int, toks[3]); d0 = parse(Int, toks[4])
    nums = parse.(Float64, toks[5:end])
    w  = nums[1:nw]
    xs = reshape(nums[nw+1:nw+d0*npt], d0, npt)   # column n = point n
    return w, xs, npt, nw, d0
end

# ---- the network, functional (no mutation, so Zygote can trace it) -----
# Same 4-8-8-1 tanh net and the same weight layout as the C++ benchmarks:
# per neuron [w_in..., bias], layer blocks consecutive.
function net(w, x1, x2, x3, x4)
    h1 = ntuple(8) do j
        p = (j - 1) * 5
        tanh(w[p+5] + w[p+1]*x1 + w[p+2]*x2 + w[p+3]*x3 + w[p+4]*x4)
    end
    h2 = ntuple(8) do j
        p = 40 + (j - 1) * 9
        tanh(w[p+9] + sum(ntuple(i -> w[p+i] * h1[i], 8)))
    end
    p = 112
    return w[p+9] + sum(ntuple(i -> w[p+i] * h2[i], 8))
end

# ---- reference values (bench/cpp/README.md; exact Taylor jet + FD) -----
const REF = Dict(
    1 => (7.539098907068e-05, -3.00304789e-04),
    2 => (5.862671447015e-08, -4.05874717e-07),
    3 => (7.890390176833e-08, -5.18918334e-07),
    4 => (7.663173332728e-10, -6.19269886e-09),
    5 => (1.158632876420e-09, -7.04080029e-09),
    6 => (5.408322663523e-11, -3.42215171e-10),
    7 => (8.175306274572e-11, -3.10379965e-10),
)

# The Val(P) function barrier matters for fairness: inside bench the
# order is a compile-time constant, so TaylorDiff specializes fully;
# only the single call from main pays a dynamic dispatch.
function bench(w, xs, npt, ::Val{P}, nrep) where {P}
    dir = [1.0, 0.0, 0.0, 0.0]
    loss(wv) = sum(map(1:npt) do n
        TaylorDiff.derivative(z -> net(wv, z[1], z[2], z[3], z[4]),
                              xs[:, n], dir, Val(P))^2
    end)

    # ---- correctness first: the run must be a real computation --------
    lv = loss(w)
    local g, t_compile
    try
        t_compile = @elapsed g = Zygote.gradient(loss, w)[1]
    catch err
        println("   Zygote could not differentiate the TaylorDiff loss:")
        println("   ", first(sprint(showerror, err), 300))
        println("   (an rrule may be missing at this order; report the")
        println("   forward loss only, and note the failure)")
        @printf("   computed loss = %.12e   (forward only)\n", lv)
        return
    end
    @printf("   computed loss = %.12e   dL/dw_1 = %+.8e\n", lv, g[1])
    lref, gref = REF[P]
    okl = abs(lv - lref) <= 1e-9 * abs(lref)
    okg = abs(g[1] - gref) <= 1e-5 * abs(gref)   # reference is central FD
    println(okl && okg ? "   check vs bench/cpp/README.md: PASS" :
            "   check vs bench/cpp/README.md: FAIL -- do not use the timing")
    @printf("   first gradient (includes compilation): %.1f s\n", t_compile)

    # ---- timing: medians of three blocks ------------------------------
    Zygote.gradient(loss, w)                       # warm
    ms = map(1:3) do _
        t = @elapsed for _ in 1:nrep
            Zygote.gradient(loss, w)
        end
        1e3 * t / nrep
    end
    @printf(" taylordiff+zygote order=%d points=%d  %10.4f ms per gradient (median of 3)\n",
            P, npt, median(ms))
end

function main()
    K    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
    nrep = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
    w, xs, npt, nw, d0 = read_wx(joinpath(@__DIR__, "wx_ref.txt"))
    bench(w, xs, npt, Val(K), nrep)
end

main()
