#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# The Kovasznay flow, an exact Navier-Stokes solution.  The shipped
# input_nn.dat is the collocation run started from fitted_weight.dat,
# the supervised fit of the exact solution, which ships with the case:
# 400 epochs of full-batch L-BFGS, reaching R2 = 0.99960, 0.99807 and
# 0.99982 for u, v and p (the figure of the paper).  Restart reads
# nn_weight.dat, so the fit is copied into place first, and any
# optimizer state left by an earlier run is removed because it belongs
# to a different point of a different run.
#
# To regenerate fitted_weight.dat itself instead (~3.5 min, 20000 epochs
# of Adam), see the block at the bottom of this file.
set -e
rm -f gd_dw.dat gd_m.dat gd_v.dat
cp fitted_weight.dat nn_weight.dat
../../build/serial.out > a.log 2>&1
echo "done.  Check the term table against the exact solution with:"
echo "  python3 ../../bench/post/check_kovasznay.py ."

# --- regenerating the fit (not run by default) ---
#   cp input_nn.dat input_collocation_backup.dat
#   cp input_fit.dat input_nn.dat
#   ../../build/serial.out
#   cp nn_weight.dat fitted_weight.dat
#   cp input_collocation_backup.dat input_nn.dat
