#!/bin/sh
# Run the full PINN benchmark suite (results referenced by the preprint).
# Usage: sh run_all.sh   (from the bench/ directory; needs ../serial.out)
for c in kdv kawahara g7 zk3 zk5 zk7 eyu10d slit opt_simple opt_ngd opt_kalman scal_closure scal_dense; do
  echo "=== $c ==="
  ( cd $c && ../../build/serial.out > a.log 2>&1 && tail -3 a.log | head -1 )
done
