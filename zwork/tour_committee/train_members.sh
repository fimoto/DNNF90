#!/bin/bash
# Train the four members of the committee.  They differ only in Rand_seed,
# which is the usual recipe: independent optima of the same problem.
for s in 101 202 303 404; do
  sed "s/^Rand_seed.*/Rand_seed    $s \//" member_template.dat > input_nn.dat
  ../../build/serial.out > /dev/null 2>&1
  mv nn_weight.dat member_$s.dat
  rm -f history_* output_* gd_*.dat data_division.log
done
echo "members trained; now run:  cp committee.dat input_nn.dat && ../../build/serial.out"
