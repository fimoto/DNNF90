#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# the electrohydrodynamic system with milder coefficients;
# 2000 epochs of the Kalman filter, several minutes

../../build/serial.out > a.log 2>&1
