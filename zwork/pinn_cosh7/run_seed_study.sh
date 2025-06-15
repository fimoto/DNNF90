#!/bin/sh
# Five seeds x four activations on the seventh-order dispersive equation
# with a hyperbolic-cosine solution, the study behind the linear
# seventh-order table of the paper.  The solution is deliberately not
# sinusoidal, so the periodic activation has no structural advantage.
#
#   sh run_seed_study.sh          # all five seeds
#   sh run_seed_study.sh 1        # the first, for a quick check
#
# Writes per_seed.csv and summary.txt.  One run is about half a minute.
set -e
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
if [ ! -x "$R/build/serial.out" ]; then
  echo "cannot find \$R/build/serial.out; build first, or set DNNF90_ROOT" >&2
  exit 1
fi
N=${1:-5}
SEEDS="11111 22222 33333 44444 55555"
ACTS="TANH SIN BESSEL BESSEL1"

i=0
for s in $SEEDS; do
  i=$((i+1))
  if [ $i -gt $N ]; then break; fi
  for a in $ACTS; do
    d=run_${s}_${a}
    rm -rf $d; mkdir $d
    cp input_nn.dat data.dat colloc.dat $d/
    sed -i "s/^Activation.*/Activation $a \//" $d/input_nn.dat
    sed -i "s/^Rand_seed.*/Rand_seed $s \//"   $d/input_nn.dat
    ( cd $d && "$R/build/serial.out" > run.log 2>&1 )
  done
done
python3 "$(dirname "$0")/summarise_linear7.py" $ACTS | tee summary.txt
