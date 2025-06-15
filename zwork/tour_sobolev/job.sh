#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# keyword tour: a Sobolev penalty from a zero high-order target

../../build/serial.out > a.log 2>&1
