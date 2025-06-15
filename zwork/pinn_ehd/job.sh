#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# The five-field electrohydrodynamic system, solved from random weights
# by full-batch L-BFGS: 0.974 -> ~1.2e-5 per point in 6000 epochs, about
# 940 s on one core, R2 >= 0.9999 in all five components.  This is the
# result reported in the paper, and input_nn.dat is a copy of
# input_cold.dat.
#
# Other input files in this directory, each run by copying it over
# input_nn.dat:
#   input_cold_ngd.dat          the same solve by the dual natural
#                               gradient on minibatches of 40 points;
#                               levels off near 1.8e-4, the minibatch
#                               being what stops it
#   input_fit.dat               supervised fit of the manufactured
#                               solution; its result is fitted_weight.dat
#   input_kalman_divergence.dat the per-pattern Kalman filter, which is
#                               measured to diverge here; kept because it
#                               is the case that divergence was
#                               characterized on (see FLOW_CASES.md)
../../build/serial.out > a.log 2>&1
