# plot.gp -- pinn_cavity.  Run `gnuplot plot.gp` after a run; writes plot.png.
#
# Column layouts:
#   data.dat            5:  x  y  u  v  p        (boundary data; lid u = 1)
#   colloc.dat          2:  x  y                 (collocation cloud, no targets)
#   output_set0002.dat  8:  x  y  u v p (network)  0 0 0 (no exact solution)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#
# The cavity has no closed-form solution, so there is no "exact" panel:
# the reference is the Ghia, Ghia & Shin (1982) centerline profiles,
# and that comparison is bench/post/cavity_compare.py.  This script
# draws the network fields and the learning curve.  The points are
# scattered, not gridded, so coloured scatter (pt 7 palette), not pm3d.

set terminal pngcairo size 1500,1200 font ",11"
set output "plot.png"

out = "output_set0002.dat"

set multiplot layout 2,2 title "Lid-driven cavity, Re = 100 (network fields; Ghia comparison: bench/post/cavity\\_compare.py)" font ",13"

set size ratio -1
set xlabel "x"; set ylabel "y"
unset key

set title "u network"
plot out u 1:2:3 with points pt 7 ps 0.6 palette notitle

set title "v network"
plot out u 1:2:4 with points pt 7 ps 0.6 palette notitle

set title "p network"
plot out u 1:2:5 with points pt 7 ps 0.6 palette notitle

# --- learning curve ---
set size noratio
set logscale y
set xlabel "epoch"; set ylabel "training cost / point"
set title "learning curve (history\\_ep*.dat col 3)"
set key top right
hist = system("ls history_ep*.dat 2>/dev/null")
plot for [f in hist] f u 1:3 with lines lw 2 title f

unset multiplot
print "wrote plot.png"
