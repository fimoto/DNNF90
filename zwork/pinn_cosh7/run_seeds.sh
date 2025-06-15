#!/bin/sh
# Paired-seed study behind the linear-dispersion table of the paper.
#
#   sh run_seeds.sh [NSEED] [ACTIVATION ...]
#
# Runs the same case once per (seed, activation) pair, all other
# settings held fixed, and writes per_seed.csv with the R^2 of every run
# against the exact solution at the collocation points.  Defaults: five
# seeds and the four activations of the table.
set -e
R=$(cd "$(dirname "$0")/../.." && pwd)
[ -x "$R/build/serial.out" ] || { echo "build first: make" >&2; exit 1; }
N=${1:-5}; shift 2>/dev/null || true
ACTS=${*:-TANH SIN BESSEL BESSEL1}
SEEDS="11111 22222 33333 44444 55555 66666 77777 88888 99999 12345"
i=0
for s in $SEEDS; do
  i=$((i+1)); [ $i -gt $N ] && break
  for a in $ACTS; do
    d="run_${s}_${a}"
    rm -rf "$d"; mkdir "$d"
    cp colloc.dat data.dat "$d"/ 2>/dev/null || true
    sed -e "s/^Rand_seed .*/Rand_seed    $s \//" \
        -e "s/^Activation .*/Activation $a \//" input_nn.dat > "$d/input_nn.dat"
    ( cd "$d" && "$R/build/serial.out" > a.log 2>&1 )
    echo "  $s $a done"
  done
done
python3 "$(dirname "$0")/summarise_seeds.py" $ACTS
