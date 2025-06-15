/* Does a work space built for a freed network get refused?
 *
 * A handle is a slot number, and a slot is reused after
 * dnnf90_net_free.  A caller that keeps a work space across the free
 * would otherwise index it against the new occupant, whose layer widths
 * may differ; the registries therefore carry a generation counter and
 * refuse a dependent object built against an earlier one.
 *
 *   cc -I lib -o handles tools/example_c_handles.c build/libdnnf90.a \
 *      -lgfortran -lm
 */
#include <stdio.h>
#include "dnnf90.h"
int main(void){
    int dA[3]={1,4,1}, dB[3]={1,32,1};
    int t = dnnf90_tabset_dense(1,2);
    int A = dnnf90_net_create_ts(t,3,dA);
    int w = dnnf90_work_create(A);
    printf("A=%d w=%d\n", A, w);
    dnnf90_net_free(A);
    int B = dnnf90_net_create_ts(t,3,dB);      /* same slot, wider layers */
    printf("B=%d (same slot as A: %s)\n", B, B==A ? "yes" : "no");
    double x[1]={0.3}, out[16];
    int rc = dnnf90_eval(B, w, x, out);        /* stale work space */
    printf("eval with the stale work space: rc=%d (expect -1)\n", rc);
    return rc==-1 ? 0 : 1;
}
