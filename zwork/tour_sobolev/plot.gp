# plot.gp -- tour_sobolev.  Run `gnuplot plot.gp` after a run; writes plot.png.
#
# Column layout of the output files (coords, then predicted, then
# target/exact; the reader of the code writes them in that order):
#   output_set0001.dat  3:  x  y (network)  y (target)
#   output_deriv_set0001.dat:  x  dy/dx (network)  (with Output_deriv 1)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#   lbfgs_history.dat   col 1 epoch, col 2 L-BFGS objective (if GD_method LBFGS)

set terminal pngcairo size 1400,550 font ",11"
set output "plot.png"
out = "output_set0001.dat"

set multiplot layout 1,2
set xlabel "x"; set ylabel "y"
set key top right
set title "target vs network"
plot out u 1:3 with points pt 6 ps 0.9 title "target", \
     out u 1:2 smooth unique with lines lw 2 title "network"

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
