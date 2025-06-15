#!/usr/bin/env python3
"""Does the distribution contain what is needed to recompute the paper?

Every figure and table of the manuscript is listed here against the case
directory or script that regenerates it.  The check is that each named
path exists and, where a runner exists, that it is executable.  It does
not rerun anything: it answers "is the input present", not "does the
number still come out".
"""
import os
import sys

import os.path
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ITEMS = [
    # (what the paper shows, what regenerates it)
    ("Fig. solution / convergence / scaling / optimizers",
     ["bench/post/make_figs.py", "bench/kdv", "bench/kawahara", "bench/g7",
      "bench/zk3", "bench/zk5", "bench/zk7", "bench/opt_ngd",
      "bench/opt_kalman", "bench/scal_closure", "bench/scal_dense"]),
    ("Fig. eyu10d",
     ["bench/post/make_fig_eyu10d.py", "bench/eyu10d"]),
    ("Fig. refine",
     ["bench/post/make_fig_refine.py", "zwork/morse_refine",
      "zwork/morse_refine/input_stage1.dat",
      "zwork/morse_refine/input_stage2.dat",
      "zwork/morse_refine/input_control.dat",
      "zwork/morse_refine/train.dat"]),
    ("Tab. bench (the shipped benchmark suite)",
     ["bench/post/make_figs.py", "bench/run_all.sh"]),
    # bench_table.tex lives with the manuscript, not the code
    ("Tab. torch (the PyTorch comparison)",
     ["bench/post/torch_pinn.py"]),
    ("Tab. speed (one weight gradient across frameworks and libraries)",
     ["bench/post/compare_frameworks.py", "tools/fwd_grad_timing.f90",
      "tools/dir_grad_timing.f90", "bench/cpp/README.md",
      "bench/cpp/codipack_bench.cpp", "bench/cpp/xad_bench.cpp",
      "bench/cpp/xad_tape_higher.cpp", "bench/cpp/adolc_grad_bench.cpp",
      "bench/julia/taylordiff_bench.jl"]),
    ("Tab. sigma (derivative growth of the activations)",
     ["tools/sigma_growth.py"]),
    ("Tab. act (order-seven-only training)",
     ["zwork/hod_4d_k7_order7", "zwork/hod_4d_k7_order7/input_nn.dat",
      "zwork/hod_4d_k7_order7/train.dat", "bench/post/hod_accuracy.py"]),
    ("Tab. k3 (third order, ten seeds, four activations)",
     ["zwork/act_seed_study", "zwork/act_seed_study/run_study.sh",
      "zwork/act_seed_study/summarise.py",
      "zwork/act_seed_study/input_nn.dat",
      "zwork/act_seed_study/train.dat"]),
    ("Tab. disp7 (the non-sinusoidal collocation case)",
     ["zwork/pinn_cosh7", "zwork/pinn_cosh7/input_nn.dat",
      "zwork/pinn_cosh7/data.dat", "zwork/pinn_cosh7/colloc.dat"]),
    ("Tab. refine (five seeds, refinement against control)",
     ["zwork/morse_refine/README.md", "bench/post/hod_accuracy.py"]),
    ("the seventh-order derivative claim on the ZK7 soliton",
     ["bench/zk7", "tools/hod_dump.f90", "bench/post/zk7_seventh.py"]),
    ("Sec. 5.4 batch and learning-rate scans",
     ["bench/kdv/input_batch20.dat", "bench/kdv/input_batch60.dat",
      "bench/kdv/input_batch135.dat", "bench/kdv/input_batch270.dat",
      "bench/kdv/input_lr1d-4.dat", "bench/kdv/input_lr3d-2.dat",
      "bench/kdv/run_scans.sh"]),
    ("Sec. 5.4 the same optimizer ordering on 2-D Poisson",
     ["zwork/pinn_poisson2d/input_nn.dat", "zwork/pinn_poisson2d/input_sgd.dat",
      "zwork/pinn_poisson2d/input_ngd.dat",
      "zwork/pinn_poisson2d_kalman/input_nn.dat",
      "zwork/pinn_poisson2d_kalman/input_decoupled.dat"]),
    ("Sec. 5.6 the residual without boundary data",
     ["zwork/pinn_kovasznay/input_no_boundary.dat"]),
    ("Sec. 5.7 the optimizer studies on the five-field system",
     ["zwork/pinn_ehd/input_cold_m8.dat", "zwork/pinn_ehd/input_adam_2d-3.dat",
      "zwork/pinn_ehd/input_adam_2d-4.dat",
      "zwork/pinn_ehd/input_cold_ngd_b40.dat",
      "zwork/pinn_ehd/input_cold_ngd_primal.dat",
      "zwork/pinn_ehd/input_kalman_colloc.dat",
      "zwork/pinn_ehd/input_kalman_colloc_lambda1.dat"]),
    ("Sec. 6 the hundredfold learning-rate scan for tanh",
     ["zwork/hod_4d_k7_order7/input_tanh_lr_x0.1.dat",
      "zwork/hod_4d_k7_order7/input_tanh_lr_x100.dat",
      "zwork/hod_4d_k7_order7/run_lr_scan.sh"]),
    ("Sec. 4.1 the forward-only cost compared with tensor_eval",
     ["tools/fwd_only_timing.f90"]),
    ("Sec. 5.1 the exact coefficients of the hierarchy",
     ["tools/check_exact_coeffs.py"]),
    ("the sinusoidal companion of disp7",
     ["zwork/pinn_disp7", "zwork/pinn_disp7/input_nn.dat"]),
]

missing = []
print("  manuscript item -> what regenerates it")
for what, paths in ITEMS:
    bad = [p for p in paths if not os.path.exists(os.path.join(R, p))]
    mark = "ok " if not bad else "MISSING"
    print("  [%s] %s" % (mark, what))
    for p in bad:
        print("           missing: %s" % p)
        missing.append((what, p))

print()
runners = ["zwork/act_seed_study/run_study.sh", "tools/negtests.sh"]
for r in runners:
    f = os.path.join(R, r)
    if os.path.exists(f):
        print("  %-38s executable: %s" % (r, os.access(f, os.X_OK)))

print()
if missing:
    print("  %d reproduction input(s) missing" % len(missing))
    sys.exit(1)
print("  every listed reproduction input is present")
print("")
print("  This is a presence check only: nothing was run and no number was")
print("  compared.  For the numerical side, run the cases and regenerate the")
print("  table and figures (docs/REPRODUCING.md), then check the values")
print("  against the manuscript; `make negtests` covers the verification")
print("  suite itself.")
