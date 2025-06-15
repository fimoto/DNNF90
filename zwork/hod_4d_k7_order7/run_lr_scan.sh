#!/bin/sh
# The hundredfold learning-rate scan of Section 6: does tanh do better on
# the seventh-derivative-only fit if its step size is retuned?  The four
# inputs span 0.1x to 100x the rate the case was tuned at, all with
# Activation TANH.  Compare the losses with the Bessel and sine rows of
# summary.txt, which use the default rate.
set -e
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -x "$R/build/serial.out" ] || { echo "build first" >&2; exit 1; }
for f in 0.1 1 10 100; do
  d=lr_x$f
  rm -rf $d; mkdir $d
  cp input_tanh_lr_x$f.dat $d/input_nn.dat
  cp train.dat $d/
  sed -i "s/^Hod_check.*/Hod_check 0 \//" $d/input_nn.dat
  ( cd $d && "$R/build/serial.out" > run.log 2>&1 )
  printf "  tanh, rate x%-5s best loss = %s\n" "$f" \
    "$(grep 'best epoch' $d/run.log | awk '{print $NF}')"
done
