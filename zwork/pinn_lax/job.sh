#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Output goes to a.log; the trainer reads input_nn.dat from this directory.
# Lax's fifth-order equation; needs the System term products

../../build/serial.out > a.log 2>&1
