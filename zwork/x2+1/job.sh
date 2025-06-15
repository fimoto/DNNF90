#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# the smallest case: y = x^2 + 1 on one input

../../build/serial.out > a.log 2>&1
