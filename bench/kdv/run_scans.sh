#!/bin/sh
# The two scans of Section 5.4, on the case of the optimizer figure.
#
#   sh run_scans.sh batch     # Num_batch 20, 60, 135, 270 at the default rate
#   sh run_scans.sh lr        # rate 1e-4 ... 3e-2 at Num_batch 20
#   sh run_scans.sh           # both
#
# Each run goes to its own directory scan_<name>/ so that nothing here is
# overwritten, and prints the floor the text quotes: the median cost over
# the last fifth of the epochs, which is a steadier number than the best
# epoch when the run is bouncing on its sampling noise.
set -e
R=${DNNF90_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -x "$R/build/serial.out" ] || { echo "build first" >&2; exit 1; }
what=${1:-both}

run_one() {          # run_one <input file>
  name=$(basename "$1" .dat | sed 's/^input_//')
  d=scan_$name
  rm -rf $d; mkdir $d
  cp "$1" $d/input_nn.dat
  cp data.dat colloc.dat $d/
  ( cd $d && "$R/build/serial.out" > a.log 2>&1 )
  python3 - "$d" <<'PY'
import sys, numpy as np
d = sys.argv[1]
h = np.loadtxt(d + "/history_ep0000000.dat")
tail = h[int(0.8 * len(h)):]
print("  %-14s floor(median of last 20%%) = %9.2e   best = %9.2e"
      % (d.replace("scan_", ""), np.median(tail[:, 2]), h[:, 2].min()))
PY
}

if [ "$what" = batch ] || [ "$what" = both ]; then
  echo "batch scan (Section 5.4: the floor falls monotonically with the batch)"
  for f in input_batch20.dat input_batch60.dat input_batch135.dat input_batch270.dat; do
    run_one $f
  done
fi
if [ "$what" = lr ] || [ "$what" = both ]; then
  echo "learning-rate scan at Num_batch 20"
  for f in input_lr1d-4.dat input_lr3d-4.dat input_lr1d-3.dat \
           input_lr3d-3.dat input_lr1d-2.dat input_lr3d-2.dat; do
    run_one $f
  done
fi
