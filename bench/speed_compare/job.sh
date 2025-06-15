#!/bin/bash
# Plain-training speed comparison (manuscript, "CPU training throughput").
# Protocol: identical architecture [1,768x5,1], tanh, float64 (neural-
# fortran runs in its native float32), one thread, full-batch SGD.
# Steady state = (t(50 epochs) - t(5 epochs))/45, which removes one-time
# costs (weight-init RNG here, interpreter start for PyTorch).  On shared
# machines interleave the runs and repeat; the clock can drift by tens of
# percent.  Run from this directory.
set -e
BIN=../../build/serial.out
[ -x $BIN ] || { echo "build first: make BLAS=1 (repo root)"; exit 1; }
run_dnnf90 () {  # $1 = epochs -> wall seconds
  sed -i "s/^Epoch.*/Epoch      $1 \//" input_nn.dat
  rm -f history_ep*.dat output* checkpoint_ep*.dat data_division.log gd_*.dat nn_weight.dat weight_new.dat checkpoint_ep*.dat
  local s=$(date +%s.%N); OPENBLAS_NUM_THREADS=1 $BIN > /dev/null 2>&1
  local e=$(date +%s.%N); echo "$e-$s" | bc
}
t50=$(run_dnnf90 50); t5=$(run_dnnf90 5)
echo "DNNF90        : $(echo "($t50-$t5)/45*1000" | bc -l | cut -c1-6) ms/epoch"
if python3 -c "import torch" 2>/dev/null; then
  OMP_NUM_THREADS=1 python3 torch_mirror.py 768 50 | sed "s/^/PyTorch CPU   : /"
else
  echo "PyTorch CPU   : (torch not installed, skipped)"
fi
if [ -x ../../build/roofline.out ]; then
  OPENBLAS_NUM_THREADS=1 ../../build/roofline.out | head -1 | sed "s/^/f64 roofline  : /"
else
  echo "f64 roofline  : build with: make roofline.out  (repo root)"
fi
cat << NF
neural-fortran: clone https://github.com/modern-fortran/neural-fortran
  (measured at commit 5e4940f), cmake -DCMAKE_BUILD_TYPE=Release, then
  gfortran -O3 -I build/include -o nf_bench.out nf_bench.f90 \\
           build/lib/libneural-fortran.a
  and take the 50-vs-5 slope of ./nf_bench.out N.
NF
sed -i "s/^Epoch.*/Epoch      50 \//" input_nn.dat
rm -f history_ep*.dat output* checkpoint_ep*.dat data_division.log gd_*.dat nn_weight.dat weight_new.dat checkpoint_ep*.dat
