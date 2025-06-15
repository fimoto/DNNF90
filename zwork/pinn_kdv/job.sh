#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# KdV soliton, the reference PINN case; ~1 min

../../build/serial.out > a.log 2>&1
