Speed-comparison kit
====================

Reproduces the CPU throughput table of the manuscript: dense network
training with no high-order machinery, 7 layers of width 768, tanh,
double precision, one thread, full batch (Num_batch 190 equals the
training split; matching the batch size matters, since the default of
10 would make the gradient work asymmetric by a factor of 19).

Files
  input_nn.dat     width-768 case for this trainer
  input_small.dat  width-10 case, where interpreter dispatch dominates
                   the PyTorch mirror
  train.dat        the data both codes fit
  job.sh           runs the protocol and prints one line per code
  torch_mirror.py  PyTorch mirror; arguments are width and epochs
  nf_bench.f90     neural-fortran mirror, in its native single precision

Build first, from the repository root:
  make BLAS=1
  make roofline.out

Then, in this directory:
  bash job.sh

Protocol
  The steady-state epoch cost is the slope (t50 - t5)/45 between
  fifty-epoch and five-epoch runs, which removes one-time costs such as
  the weight-initialization draws and the PyTorch interpreter start.
  Interleave the codes and repeat the set: on a shared machine the clock
  drifts by tens of percent, and only ratios measured in the same phase
  are meaningful.

Reference toolchain for the numbers in the manuscript
  gfortran 13.3.0, OpenBLAS 0.3.26, PyTorch 2.13 (CPU),
  neural-fortran commit 5e4940f, one thread throughout.
