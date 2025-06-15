#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# K=7 restricted to the closure of the seeds in
# alpha_seeds.dat, which is what the residual would ask for

../../build/serial.out > a.log 2>&1
