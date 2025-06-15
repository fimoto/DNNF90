/* tools/example_c.c - DNNF90 called from C, the way n2p2 or a LAMMPS
 * pair style would call it.  (MIT License.)
 *
 * Run in a trained benchmark directory (Restart not needed; the weight
 * file is read directly), for example:
 *     cd bench/kdv && ../../c_example.out
 *
 * It rebuilds the same closure tables the KdV network was trained with,
 * loads the trained network, evaluates value and derivatives at three
 * points, prints the KdV residual u_t + 3 u u_x + u_xxx, and if
 * c_ref.dat exists (written by gen_c_ref.out) compares bitwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dnnf90.h"

int main(void)
{
    /* the KdV residual seeds: u_t, u_x, u_xxx  ->  closure has 5 slots */
    const int seeds[6] = { 0,1,  1,0,  3,0 };
    if (dnnf90_tables_init_closure(2, 3, 3, seeds) != 0) return 1;
    const int n = dnnf90_nderiv();
    printf("carried slots: %d\n", n);

    const int nid = dnnf90_net_load("nn_weight.dat");
    if (nid <= 0) { printf("net_load failed (%d)\n", nid); return 1; }
    const int wid = dnnf90_work_create(nid);

    /* locate u, u_x, u_t, u_xxx among the slots */
    int iu=0, iux=0, iut=0, iu3=0, a[2];
    for (int ia = 1; ia <= n; ++ia) {
        dnnf90_alpha(ia, a);
        if (a[0]==0 && a[1]==0) iu  = ia;
        if (a[0]==1 && a[1]==0) iux = ia;
        if (a[0]==0 && a[1]==1) iut = ia;
        if (a[0]==3 && a[1]==0) iu3 = ia;
    }

    double x[2], t[64];
    printf("    x     t         u        u_x        u_t      u_xxx   residual\n");
    for (int ip = 0; ip < 3; ++ip) {
        x[0] = -2.0 + 2.0*ip;  x[1] = 0.5;
        dnnf90_eval(nid, wid, x, t);
        const double res = t[iut-1] + 3.0*t[iu-1]*t[iux-1] + t[iu3-1];
        printf("%6.2f %5.2f %10.3e %10.3e %10.3e %10.3e %10.2e\n",
               x[0], x[1], t[iu-1], t[iux-1], t[iut-1], t[iu3-1], res);
    }

    /* bitwise check against the Fortran reference, if present */
    FILE *f = fopen("c_ref.dat", "r");
    if (f) {
        double dmax = 0.0;
        int np; if (fscanf(f, "%d", &np) != 1) np = 0;
        for (int ip = 0; ip < np; ++ip) {
            if (fscanf(f, "%lf %lf", &x[0], &x[1]) != 2) break;
            double tr[64];
            for (int ia = 0; ia < n; ++ia)
                if (fscanf(f, "%lf", &tr[ia]) != 1) break;
            dnnf90_eval(nid, wid, x, t);
            for (int ia = 0; ia < n; ++ia) {
                const double d = fabs(t[ia] - tr[ia]);
                if (d > dmax) dmax = d;
            }
        }
        fclose(f);
        printf("max |C - Fortran| over reference points: %.3e  %s\n",
               dmax, dmax == 0.0 ? "(bitwise identical)" : "(MISMATCH)");
    }

    dnnf90_work_free(wid);
    dnnf90_net_free(nid);
    return 0;
}
