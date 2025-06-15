/* tools/example_c_train.c - training driven entirely from C through the
 * ABI: the shape an n2p2 based trainer would use, where the host owns
 * the descriptors and the loss and DNNF90 owns network, gradient and
 * optimizer.  (MIT License.)  Run in bench/kdv.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dnnf90.h"

int main(void)
{
    const int seeds[6] = { 0,1, 1,0, 3,0 };
    dnnf90_tables_init_closure(2, 3, 3, seeds);
    const int n = dnnf90_nderiv();

    /* small fresh network trained from C on a toy target u* = x1*x2 */
    const int dims[3] = { 2, 8, 1 };
    const int nid = dnnf90_net_create(3, dims);
    /* deterministic tiny init through set_layer */
    double row[64];
    for (int l = 2; l <= 3; ++l) {
        const int nd  = (l == 2) ? 8 : 1;
        const int ndm = (l == 2) ? 2 : 8;
        for (int j = 0; j < nd; ++j)
            for (int i = 0; i <= ndm; ++i)
                row[j*(ndm+1)+i] = 0.1*sin(1.7*l + 0.9*(j+1) + 0.3*i) + 0.05;
        dnnf90_net_set_layer(nid, l, row);
    }
    const int wid = dnnf90_work_create(nid);
    const int tid = dnnf90_twork_create(nid);
    const int gid = dnnf90_grad_create(nid);

    const int NPT = 200;
    double x[2], t[64], seedv[64], loss = 0.0, loss0 = -1.0;
    for (int ep = 1; ep <= 400; ++ep) {
        dnnf90_grad_zero(gid);
        loss = 0.0;
        for (int p = 0; p < NPT; ++p) {
            x[0] = -1.0 + 2.0*((7*p+3) % 100)/99.0;
            x[1] = -1.0 + 2.0*((13*p+5) % 100)/99.0;
            dnnf90_eval(nid, wid, x, t);
            const double r = t[0] - x[0]*x[1];      /* data residual */
            loss += 0.5*r*r;
            for (int ia = 0; ia < n; ++ia) seedv[ia] = 0.0;
            seedv[0] = r;                           /* dL/du seed */
            dnnf90_grad_point(nid, tid, x, seedv, gid);
        }
        if (ep == 1) loss0 = loss/NPT;
        dnnf90_adam_step(nid, gid, 3e-3, 0.9, 0.999, 1e-8, NPT, ep);
    }
    printf("C-driven training: loss %.3e -> %.3e over 400 epochs  %s\n",
           loss0, loss/NPT, (loss/NPT < 0.02*loss0) ? "(converging)" : "(NOT converging)");

    dnnf90_grad_free(gid); dnnf90_twork_free(tid);
    dnnf90_work_free(wid); dnnf90_net_free(nid);
    return 0;
}
