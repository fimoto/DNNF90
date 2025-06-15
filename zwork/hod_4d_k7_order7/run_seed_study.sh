#!/bin/sh
# Eight seeds x four activations on the seventh-derivative-only fit.
# Each run teaches only |alpha| = 7 and leaves the value and every lower
# order free, so what is compared is how well each activation lets the
# network carry the top order on its own.
#
#   sh run_seed_study.sh          # all eight seeds
#   sh run_seed_study.sh 2        # the first two, for a quick check
#
# Writes per_seed.csv and summary.txt.  One run is about a minute.
set -e
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ ! -x "$R/build/serial.out" ]; then
  echo "cannot find \$R/build/serial.out; build first, or set DNNF90_ROOT" >&2
  exit 1
fi
N=${1:-8}
SEEDS="11111 22222 33333 44444 55555 66666 77777 88888"
ACTS="TANH SIN BESSEL BESSEL1"

i=0
for s in $SEEDS; do
  i=$((i+1))
  if [ $i -gt $N ]; then break; fi
  for a in $ACTS; do
    d=run_${s}_${a}
    rm -rf $d; mkdir $d
    cp input_nn.dat train.dat $d/
    sed -i "s/^Activation.*/Activation $a \//"  $d/input_nn.dat
    sed -i "s/^Rand_seed.*/Rand_seed $s \//"    $d/input_nn.dat
    sed -i "s/^Hod_check.*/Hod_check 0 \//"     $d/input_nn.dat
    ( cd $d && "$R/build/serial.out" > run.log 2>&1 )
  done
done
python3 "$(dirname "$0")/summarise_seventh.py" $ACTS | tee summary.txt
