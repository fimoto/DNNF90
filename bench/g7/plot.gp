# plot.gp -- g7.  Run `gnuplot plot.gp` after a run; writes plot.png.
#
# Column layout of the output files (coords, then predicted, then
# target/exact; the reader of the code writes them in that order):
#   output_set0002.dat  4:  x1  x2  u (network)  u* (exact)
#                       (axis 2 is time for the evolution cases:
#                        Time_axis defaults to the last axis)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#   lbfgs_history.dat   col 1 epoch, col 2 L-BFGS objective (if GD_method LBFGS)
#
# Three things, per the case notes: the exact solution, the learning
# curve on a log axis, and the network solution ON THE SAME COLOUR
# SCALE as the exact one, so a small error looks small instead of being
# amplified by its own range.  The points are scattered, not gridded,
# so coloured scatter (pt 7 palette), not pm3d.

set terminal pngcairo size 1400,1100 font ",11"
set output "plot.png"
out = "output_set0002.dat"

set multiplot layout 2,2 title "exact vs network" font ",14"
set size ratio -1
set xlabel "x1"; set ylabel "x2"
unset key

stats out u 4 nooutput
set cbrange [STATS_min:STATS_max]
set title "u exact"
plot out u 1:2:4 with points pt 7 ps 0.6 palette notitle
set title "u network (same scale)"
plot out u 1:2:3 with points pt 7 ps 0.6 palette notitle

# --- learning curve: every history file, so a restarted run shows both halves ---
set size noratio
unset cbrange
set logscale y
set xlabel "epoch"; set ylabel "training cost / point"
set title "learning curve (history\\_ep*.dat col 3)"
set key top right
hist = system("ls history_ep*.dat 2>/dev/null")
plot for [f in hist] f u 1:3 with lines lw 2 title f

# --- L-BFGS objective, if that method was used ---
lb = system("ls lbfgs_history.dat 2>/dev/null")
if (strlen(lb) > 0) {
    set title "L-BFGS line-search objective (lbfgs\\_history.dat)"
    set xlabel "epoch"; set ylabel "objective"
    plot "lbfgs_history.dat" u 1:2 with lines lw 2 notitle
} else {
    set title "lbfgs\\_history.dat not present (GD\\_method is not LBFGS)"
    unset logscale y
    set key off
    plot [-1:1][-1:1] 1/0 notitle
}

unset multiplot
print "wrote plot.png"
