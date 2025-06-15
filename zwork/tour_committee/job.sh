#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Two steps: train the four members (they differ only in Rand_seed), then
# run the COMMITTEE task, which reads the member weight files and reports
# the mean and spread of every carried slot.  The member weights ship
# with the case, so this reproduces them from scratch.
# The shipped input_nn.dat is a copy of committee.dat, so that this
# directory behaves like every other case if the trainer is run in it
# directly: the member weight files ship with the case, so the COMMITTEE
# task runs as it stands.  job.sh retrains the members first.
set -e
./train_members.sh
cp committee.dat input_nn.dat
../../build/serial.out > a.log 2>&1
echo "done.  Committee mean and spread are in the output files; plot.gp draws them."
