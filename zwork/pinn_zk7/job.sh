#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# seventh-order extended Zakharov-Kuznetsov in (3+1)D;
# the headline case of the paper, ~15 s

../../build/serial.out > a.log 2>&1
