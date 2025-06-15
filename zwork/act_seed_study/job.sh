#!/bin/sh
# Run this case as it ships:   sh job.sh    (or ./job.sh)
# The paired activation study: one case, several activations, several
# seeds, with the seed differences cancelled.  Defaults to the study
# reported in the paper (ten seeds, four activations); run_study.sh takes
# the seed count and the activation list if you want a smaller one.
set -e
./run_study.sh 10 TANH SIN BESSEL BESSEL1
echo "done.  Summary table: summary.txt (also printed above)."
