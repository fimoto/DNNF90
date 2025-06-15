# plot.gp -- pinn_ehd (also usable in pinn_ehd_mild).  Run `gnuplot
# plot.gp` after a run; writes plot.png.
#
# Column layouts (components: 1 phi, 2 rho, 3 u, 4 v, 5 p):
#   data.dat            7:  x  y  phi rho u v p                  (boundary data)
#   colloc.dat         12:  x  y  S1..S5 (sources, Sys_src)  phi* rho* u* v* p*
#   output_set0002.dat 12:  x  y  phi rho u v p (network)  phi* rho* u* v* p* (exact)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#   lbfgs_history.dat   col 1 epoch, col 2 L-BFGS objective (if GD_method LBFGS)
#
# The solution is manufactured, so every component has an exact panel.
# The network panel of each component is drawn ON THE SAME COLOUR SCALE
# as its exact one, so a small error looks small instead of being
# amplified by its own range.  The points are scattered, not gridded,
# so coloured scatter (pt 7 palette), not pm3d.

set terminal pngcairo size 1500,2300 font ",11"
set output "plot.png"

out = "output_set0002.dat"
names = "phi rho u v p"

set multiplot layout 6,2 title "Electrohydrodynamic system: exact vs network" font ",14"

set size ratio -1
set xlabel "x"; set ylabel "y"
unset key

do for [i=1:5] {
    nm = word(names, i)
    stats out u (column(7+i)) nooutput
    set cbrange [STATS_min:STATS_max]
    set title nm . " exact"
    plot out u 1:2:(column(7+i)) with points pt 7 ps 0.5 palette notitle
    set title nm . " network (same scale)"
    plot out u 1:2:(column(2+i)) with points pt 7 ps 0.5 palette notitle
}

# --- learning curve ---
set size noratio
unset cbrange
set logscale y
set xlabel "epoch"; set ylabel "training cost / point"
set title "learning curve (history\\_ep*.dat col 3)"
set key top right
hist = system("ls history_ep*.dat 2>/dev/null")
plot for [f in hist] f u 1:3 with lines lw 2 title f

# --- L-BFGS objective, if that method was used (the one rule that descends this system) ---
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
