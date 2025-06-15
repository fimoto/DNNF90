#!/bin/bash
# Paired activation comparison over seeds.
#
# The four runs of a seed share the data, the network shape, the schedule
# and the random draw; only the activation differs.  The case is the one
# shipped as zwork/hod_4d_k3, which was set up for tanh, so any tuning in
# it favours tanh.
#
#   ./run_study.sh [n_seeds] [activations ...]
#
# Writes one directory per (seed, activation) and a summary table.  With
# the defaults it is 20 runs of about two seconds each.
set -e
# the repository root: two levels up from this script, unless DNNF90_ROOT
# says otherwise (useful when the study is copied elsewhere)
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ ! -x "$R/build/serial.out" ]; then
  echo "cannot find \$R/build/serial.out; build first, or set DNNF90_ROOT" >&2
  exit 1
fi
N=${1:-10}; shift || true
ACTS=${*:-"TANH SIN"}
SEEDS="11111 22222 33333 44444 55555 66666 77777 88888 99999 12345"

i=0
for s in $SEEDS; do
  i=$((i+1)); [ $i -gt $N ] && break
  for a in $ACTS; do
    d=run_${s}_${a}
    rm -rf $d; mkdir $d
    cp input_nn.dat train.dat $d/
    sed -i "s/^Activation.*/Activation $a \//"  $d/input_nn.dat
    sed -i "s/^Rand_seed.*/Rand_seed $s \//"    $d/input_nn.dat
    sed -i "s/^Hod_check.*/Hod_check 0 \//"     $d/input_nn.dat
    ( cd $d && $R/build/serial.out > run.log 2>&1 )
  done
done
python3 "$(dirname "$0")/summarise.py" $ACTS | tee summary.txt
