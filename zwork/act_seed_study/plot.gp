# plot.gp -- act_seed_study.  Run `gnuplot plot.gp` after a run; writes plot.png.
#
# Column layout of the output files (coords, then predicted, then
# target/exact; the reader of the code writes them in that order):
#   output_set0001.dat  6:  x(1:4)  prediction(1:1)  target(1:1)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#   lbfgs_history.dat   col 1 epoch, col 2 L-BFGS objective (if GD_method LBFGS)
#
# The inputs are 4-dimensional, so a field cannot be drawn as a
# plane: the honest picture is the parity plot, network against
# target, where a perfect fit is the diagonal.

set terminal pngcairo size 1400,560 font ",11"
set output "plot.png"
out = "output_set0001.dat"

set multiplot layout 1,2
set key off

stats out u 6 nooutput
lo = STATS_min; hi = STATS_max
set xrange [lo:hi]; set yrange [lo:hi]
set size ratio 1
set xlabel "target"; set ylabel "network"
set title "parity, component 1 of 1"
plot x with lines lc rgb "gray" lw 1 notitle, \
     out u 6:5 with points pt 7 ps 0.5 notitle
unset xrange; unset yrange

# --- learning curve: every history file, so a restarted run shows both halves ---
set size noratio
unset cbrange
set logscale y
set xlabel "epoch"; set ylabel "training cost / point"
set title "learning curve (history\\_ep*.dat col 3)"
set key top right
hist = system("ls history_ep*.dat 2>/dev/null")
plot for [f in hist] f u 1:3 with lines lw 2 title f

unset multiplot
print "wrote plot.png"
