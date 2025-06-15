# Refining a force field with high-order derivatives

A Morse chain of six atoms, one dimension, with the atomic positions as
the network input, so the carried multi-index slots are the force
constants themselves and no descriptor stands between the engine and the
quantity of interest. The targets are analytic: in one dimension
`dr/dx = ±1` and every higher derivative of `r` vanishes, so

    d^alpha E = sum_pairs V^(|alpha|)(r) s^|alpha| (-1)^(alpha_j)

over the pairs that carry the whole multi-index, with

    V^(n)(r) = 2 D a^n (-1)^n e (2^(n-1) e - 1),   e = exp(-a(r-r0)),

checked against numerical differentiation to 1e-29.

## The question

A machine-learned force field is fitted to energies and forces, because
that is what a first-principles calculation gives up cheaply. The
anharmonic phonon quantities are higher derivatives: the harmonic force
constants are the second, the cubic anharmonicity that sets three-phonon
scattering and the thermal conductivity is the third, the quartic terms
that shift frequencies with temperature are the fourth. Does a force
field validated on forces know anything about them, and if not, can it be
refined?

## Running

    cp input_stage1.dat input_nn.dat && ../../build/serial.out
    cp nn_weight.dat stage1_weights.dat

    rm -f gd_*.dat                  # see the note below
    cp input_stage2.dat input_nn.dat && cp stage1_weights.dat nn_weight.dat
    ../../build/serial.out          # the refinement

    rm -f gd_*.dat
    cp input_control.dat input_nn.dat && cp stage1_weights.dat nn_weight.dat
    ../../build/serial.out          # the control: same epochs, no high orders

The `rm -f gd_*.dat` matters. A restart reads the optimizer state as well
as the weights, and the run refuses to start if the two are from
different epochs rather than silently mixing them. Both stages here start
from the stage-one weights, so the optimizer log of the previous stage
has to go; the moment estimates of Adam are rebuilt in a few epochs.

Then `python3 ../../bench/post/hod_accuracy.py .` on each.

## What it shows

Relative error against the analytic force constants, one seed with tanh:

| stage | forces | harmonic | cubic | quartic |
|---|---|---|---|---|
| forces only | 0.132 | 0.218 | **0.838** | **1.336** |
| control, +2000 epochs, forces only | 0.094 | 0.156 | **0.655** | **1.399** |
| refined, +2000 epochs, orders 2-4 added | 0.039 | 0.022 | **0.033** | **0.050** |

Medians over four seeds, tanh. Across the seven completed (seed,
activation) pairs the refinement beats the control at every order and on
every pair, a sign test at p = 0.008, by a median factor of 9 at second
order, 24 at third and 19 at fourth.

Two things. A force field with a force error of three percent, which
passes the usual validation without comment, has a quartic error above
one: no information at all. And the refinement recovers it, by a factor
of twenty in the cubic and thirty in the quartic, while the control shows
that the extra epochs alone do nothing.

The loss weights are `lambda_p = 1/||y_p||^2`, each order normalised by
its own scale. That is not cosmetic: the quartic targets are fifty times
the size of the forces, and an equally weighted loss is dominated by them
to the point that the forces are never fitted. The energy is not taught
at all, since once the derivatives are constrained it is determined up to
the additive constant the physics does not fix.
