#!/bin/sh
# Reproduce the seventh-order activation table: eight seeds, four
# activations, one directory per pair, and a per-seed CSV.
#
#   ./run_seeds.sh [n_seeds] [activations ...]
#
# The manuscript reports the median and the range over the seeds and a
# sign test against tanh, so the individual runs have to be available;
# per_seed.csv is what those figures are computed from.
set -e
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ ! -x "$R/build/serial.out" ]; then
  echo "cannot find \$R/build/serial.out; build first, or set DNNF90_ROOT" >&2
  exit 1
fi
N=${1:-8}; shift 2>/dev/null || true
ACTS=${*:-"TANH SIN BESSEL BESSEL1"}
SEEDS="11111 22222 33333 44444 55555 66666 77777 88888"

i=0
for s in $SEEDS; do
  i=$((i+1)); [ $i -gt $N ] && break
  for a in $ACTS; do
    d=run_${s}_${a}
    rm -rf $d; mkdir $d
    cp input_nn.dat train.dat $d/ 2>/dev/null || cp input_nn.dat $d/
    for f in *.dat; do [ -f "$f" ] && cp "$f" $d/ 2>/dev/null || true; done
    sed -i "s/^Rand_seed .*/Rand_seed    $s \//" $d/input_nn.dat
    sed -i "s/^Activation .*/Activation $a \//" $d/input_nn.dat
    ( cd $d && rm -f gd_*.dat nn_weight.dat && "$R/build/serial.out" > a.log 2>&1 )
    echo "  $s $a done"
  done
done
python3 "$(dirname "$0")/summarise_seeds.py" $ACTS | tee summary.txt
