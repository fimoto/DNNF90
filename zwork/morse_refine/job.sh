#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# Three runs in sequence (see README.md): the forces-only fit, then the
# refinement that adds orders 2-4, then the control that spends the same
# extra epochs on forces alone.  Each stage writes its own log.
#
# The rm -f gd_*.dat matters: a restart reads the optimizer state as well
# as the weights and refuses to mix states from different epochs.  Both
# later stages start from the stage-one weights, so the previous
# optimizer log has to go.
# The shipped input_nn.dat is a copy of input_stage1.dat, so that this
# directory behaves like every other case if the trainer is run in it
# directly; job.sh overwrites it stage by stage in any event.
set -e
S=../../build/serial.out

cp input_stage1.dat input_nn.dat
$S > a_stage1.log 2>&1
cp nn_weight.dat stage1_weights.dat

rm -f gd_*.dat
cp input_stage2.dat input_nn.dat
cp stage1_weights.dat nn_weight.dat
$S > a_stage2_refined.log 2>&1
cp nn_weight.dat stage2_weights.dat

rm -f gd_*.dat
cp input_control.dat input_nn.dat
cp stage1_weights.dat nn_weight.dat
$S > a_stage3_control.log 2>&1

echo "done.  Accuracy by derivative order:"
echo "  python3 ../../bench/post/hod_accuracy.py ."
