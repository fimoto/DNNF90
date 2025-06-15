#!/bin/bash
# Property and rejection tests: the checks a finite-difference comparison
# cannot make.  Each case must stop with the quoted message, or (for the
# property tests) must produce the stated exact value.
#   usage:  tools/negtests.sh          (from the repository root)
R=$(cd "$(dirname "$0")/.." && pwd)
# the two example binaries these tests need
make -C $R fdcheck.out libverify_example.out product_adj_example.out multiout_adj_example.out batch_act_example.out act_roundtrip_example.out api_lifecycle_example.out lifecycle_switch_example.out c_handles.out > /dev/null 2>&1
BIN=$R/build/serial.out
# The suite is also runnable directly, so the binaries it needs are built
# here when they are absent rather than assumed to be present.
for t in serial.out hod_ff_example.out; do
  [ -x "$R/build/$t" ] || ( cd $R && make $t > /dev/null 2>&1 ) \
    || { echo "cannot build $t"; exit 1; }
done
W=$(mktemp -d); pass=0; fail=0
say() { if [ "$1" = ok ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "  FAIL: $2"; fi; }

reject() { # reject <case-dir> <sed-edit> <expected message>
  d=$W/r$RANDOM; rm -rf $d; cp -r "$1" $d; ( cd $d && eval "$2" \
    && timeout 200 $BIN 2>&1 | grep -q "$3" ) && say ok || say no "$3"
}

echo "rejection tests"
reject $R/zwork/x2+1 "sed -i '/^GD_param/d' input_nn.dat"            "no optimizer step"
reject $R/zwork/x2+1 "python3 - <<'P'
import re;s=open('input_nn.dat').read()
open('input_nn.dat','w').write(re.sub(r'Nlayer.*?\n(?:\s*\d+\s*/[^\n]*\n)+','',s,count=1))
P"                                                                    "Nlayer is required"
reject $R/zwork/x2+1 "sed -i 's/^Init_w.*/Init_w Banana/' input_nn.dat"  "unknown Init_weight"
reject $R/zwork/x2+1 "sed -i 's/^Activation.*/Activation ELU \//' input_nn.dat" "Activation:"
reject $R/zwork/x2+1 "sed -i 's/^GD_method.*/GD_method Adam/;s/^GD_param.*/GD_param 0.01d0 0.9d0 0.999d0 0.d0 0.d0 \//' input_nn.dat" "must be positive"
reject $R/zwork/x2+1 "printf 'Task BANANA /\n' >> input_nn.dat"      "unknown value"
reject $R/zwork/x2+1 "printf 'Fit 1 /\n' >> input_nn.dat"            "has been renamed"
reject $R/bench/kdv  "python3 -c \"
s=open('input_nn.dat').read()
open('input_nn.dat','w').write(s.replace('Residual 3 /','Residual 70 /'+chr(10)+('LIN 1.d0 0 1 /'+chr(10))*67))\"" "at most"
reject $R/bench/zk7  "printf 'Time_axis 9 /\n' >> input_nn.dat"      "must lie in"
reject $R/bench/zk7  "printf 'Time_axis 1 /\n' >> input_nn.dat"      "is the time axis"
reject $R/zwork/pinn_poisson2d_kalman "sed -i 's/^GD_param.*/GD_param 0.d0 0.98d0 0.9995d0 0.d0 0.d0 \//' input_nn.dat" "Kalman needs p1"
reject $R/zwork/tour_committee "cp committee.dat input_nn.dat && sed -i 's/^Committee 4 /Committee 1 /' input_nn.dat" "at least 2 members"
reject $R/zwork/tour_committee "cp committee.dat input_nn.dat && rm -f member_303.dat" "cannot open"

# a record with a missing column must be caught in every loss form, not
# silently completed from the next line
reject $R/zwork/x2+1 "python3 -c \"
ls=open('train.dat').read().split(chr(10)); ls[0]=ls[0].split()[0]
open('train.dat','w').write(chr(10).join(ls))\"" "a line is missing a value"
reject $R/zwork/pinn_poisson2d "python3 -c \"
ls=open('colloc.dat').read().split(chr(10)); ls[0]=' '.join(ls[0].split()[:3])
open('colloc.dat','w').write(chr(10).join(ls))\"" "a line is missing a value"
reject $R/zwork/hod_4d_k3 "python3 -c \"
ls=open('train.dat').read().split(chr(10)); ls[0]=' '.join(ls[0].split()[:20])
open('train.dat','w').write(chr(10).join(ls))\"" "a line is missing a value"

echo "property tests"
# identical members must show exactly zero spread: the textbook variance
# formula returns noise or a clipped zero here instead
d=$W/c; rm -rf $d; cp -r $R/zwork/tour_committee $d
( cd $d && for k in 2 3 4; do cp member_101.dat member_dup$k.dat; done
  sed 's/member_202.dat/member_dup2.dat/;s/member_303.dat/member_dup3.dat/;s/member_404.dat/member_dup4.dat/' \
      committee.dat > input_nn.dat
  timeout 300 $BIN 2>&1 | awk '/spread over/{f=1} f&&/E[+-]/{if ($4+0!=0 || $5+0!=0) bad=1} END{exit bad}' ) \
  && say ok || say no "identical committee members give exactly zero spread"

# The generator must reproduce the library enumeration exactly: the dense
# set and a seed closure.  If these drift apart, a HOD_DATA file prepared
# from the script would silently have its columns permuted.
for spec in "4 --k 3|$R/zwork/hod_4d_k3" \
            "4 --seeds $R/zwork/hod_4d_k7_active/alpha_seeds.dat|$R/zwork/hod_4d_k7_active"; do
  arg="${spec%%|*}"; dir="${spec##*|}"
  d=$W/a$RANDOM; rm -rf $d; cp -r "$dir" $d
  ( cd $d \
    && sed -i 's/^Epoch.*/Epoch 2 \//' input_nn.dat \
    && sed -i 's/^Hod_check.*/Hod_check 0 \//' input_nn.dat \
    && timeout 300 $BIN > /dev/null 2>&1 \
    && python3 $R/tools/alpha_order.py --d0 $arg > py.dat \
    && cmp -s py.dat hod_alpha_order.dat ) \
    && say ok || say no "alpha_order.py reproduces hod_alpha_order.dat ($arg)"
done

# The force-field example checks its own hand-written derivatives; the
# suite makes that a regression rather than something a reader must run.
d=$W/ff; rm -rf $d; mkdir -p $d
( cd $d && timeout 2000 $R/build/hod_ff_example.out > ff.log 2>&1 \
  && grep -q "ALL PASSED" ff.log ) \
  && say ok || say no "hod_ff_example chain-rule checks pass"

# the library self-verification must actually fail when it should, so it
# is run here for its exit status rather than for its printout
( cd $W && $R/build/libverify_example.out > /dev/null 2>&1 ) \
  && say ok || say no "libverify_example exits zero when the library is sound"

# the gradient check is the one the documentation points at first, so its
# exit status has to mean something
( cd $W && $R/build/fdcheck.out > /dev/null 2>&1 ) \
  && say ok || say no "fdcheck.out exits zero when the gradients are sound"

# the K=7, 330-slot golden regression, run where its file lives
( cd $R/zwork/hod_4d_k7 && $R/build/serial.out 2>/dev/null \
    | grep -q "\[REF\] golden regression: 330 values" ) \
  && say ok || say no "the golden regression compares all 330 slots at K=7"
( cd $R/zwork/hod_4d_k7 && $R/build/serial.out 2>/dev/null | grep -q "\[REF\] passed" ) \
  && say ok || say no "the golden regression passes at K=7"

# a weight file trained with one activation must not be restarted under
# another, and a shape mismatch must not be read as if it fitted
( cd $W && cp -r $R/zwork/tour_derivfit hdr && cd hdr \
    && sed -i 's/^Activation .*/Activation BESSEL \//; s/^Epoch .*/Epoch      3 \//' input_nn.dat \
    && rm -f gd_*.dat nn_weight.dat && $R/build/serial.out > /dev/null 2>&1 \
    && sed -i 's/^Activation .*/Activation TANH \//; s/^Restart .*/Restart      1 \//' input_nn.dat \
    && rm -f gd_*.dat \
    && $R/build/serial.out 2>&1 | grep -q "activation code" ) \
  && say ok || say no "restarting non-tanh weights under tanh is refused"

( cd $W && $R/build/act_roundtrip_example.out 2>/dev/null | grep -q passed ) \
  && say ok || say no "a weight file survives save and load for every activation"

( cd $W && $R/build/c_handles.out > /dev/null 2>&1 ) \
  && say ok || say no "a work space built for a freed network is refused, not reinterpreted"

# one process, two different configurations, both directions
( cd $W && mkdir -p lcw && cp -r $R/zwork/pinn_taylorgreen lcw/lc_sys \
    && cp -r $R/zwork/x2+1 lcw/lc_plain \
    && sed -i 's/^Epoch .*/Epoch      2 \//' lcw/lc_sys/input_nn.dat \
    && sed -i 's/^Epoch .*/Epoch      2 \//' lcw/lc_plain/input_nn.dat \
    && cd lcw && $R/build/lifecycle_switch_example.out 2>/dev/null \
    | grep -q passed ) \
  && say ok || say no "init/free/init across different configurations in one process"

( cd $R/zwork/x2+1 && $R/build/api_lifecycle_example.out 2>/dev/null \
    | grep -q passed ) \
  && say ok || say no "the API derivative table is the same before and after the first evaluation"

( cd $W && $R/build/batch_act_example.out 2>/dev/null | grep -q passed ) \
  && say ok || say no "the batched value path matches the per-point path for every activation"

( cd $W && $R/build/product_adj_example.out 2>/dev/null | grep -q passed ) \
  && say ok || say no "the product-rule adjoint agrees with central differences for one to four factors"

( cd $W && $R/build/multiout_adj_example.out 2>/dev/null | grep -q "vs FD" ) \
  && say ok || say no "the multi-output adjoint returns a weight gradient"

# the coupled-system residual and its adjoint, on a problem with an exact
# solution: the residual of the Taylor-Green vortex must vanish and the
# seed must reproduce differences of the loss
( cd $W && $R/build/taylorgreen_example.out 2>/dev/null | \
  awk '/exact Taylor-Green/{for(i=NF-2;i<=NF;i++) if ($i+0>1e-12 || $i+0<-1e-12) bad=1}
       /adjoint vs FD/{if ($NF+0 > 1e-7) bad=1}
       END{exit bad?1:0}' ) \
  && say ok || say no "Taylor-Green system residual and adjoint"

echo "$pass passed, $fail failed"
rm -rf $W
[ $fail -eq 0 ]
