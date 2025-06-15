# plot.gp -- tour_committee.  Run `gnuplot plot.gp` after a run; writes plot.png.
#
# Column layout of the output files (coords, then predicted, then
# target/exact; the reader of the code writes them in that order):
#   output_committee_set0001.dat  5:  x  mean(value)  mean(du/dx)
#                                     std(value)  std(du/dx)
#   (Task COMMITTEE writes the mean and the sample standard deviation
#    of every carried slot; with Output_deriv 1 the first-order set
#    is carried, so the spread of dN/dx is reported next to the
#    spread of the value.  No history: the task does not train.)
#   history_ep*.dat     col 1 epoch, col 3 training cost per point
#   lbfgs_history.dat   col 1 epoch, col 2 L-BFGS objective (if GD_method LBFGS)

set terminal pngcairo size 1400,550 font ",11"
set output "plot.png"
out = "output_committee_set0001.dat"

set multiplot layout 1,2
set xlabel "x"
set key top right

set ylabel "N(x)"
set title "committee mean {/Symbol \261} std of the value"
plot out u 1:2:4 with yerrorbars pt 7 ps 0.4 lc rgb "#88aa3333" title "mean {/Symbol \261} std", \
     out u 1:2 smooth unique with lines lw 2 lc rgb "dark-red" title "mean"

set ylabel "dN/dx"
set title "committee mean {/Symbol \261} std of the derivative"
plot out u 1:3:5 with yerrorbars pt 7 ps 0.4 lc rgb "#883333aa" title "mean {/Symbol \261} std", \
     out u 1:3 smooth unique with lines lw 2 lc rgb "navy" title "mean"

unset multiplot
print "wrote plot.png"
