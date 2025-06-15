/* dnnf90.h - C interface of the DNNF90 instance based engine.
 * (MIT License; see LICENSE at the repository root.)
 *
 * Link against libdnnf90.a (or .so) built by "make lib" / "make shared".
 * Handles are > 0; negative return values are errors; 0 is success for
 * routines without a handle result.  One work handle per thread makes
 * evaluation and gradient accumulation thread safe.
 */
#ifndef DNNF90_H
#define DNNF90_H
#ifdef __cplusplus
extern "C" {
#endif

/* shared read-only tables (call one of these once, first) */
int dnnf90_tables_init_dense(int d0, int k);
int dnnf90_tables_init_closure(int d0, int k, int nseed, const int *seeds);
int dnnf90_nderiv(void);
int dnnf90_alpha(int ia, int *a);

/* networks */
/* independent table sets (per-species descriptor count and order) */
int dnnf90_tabset_dense(int d0, int k);
int dnnf90_tabset_closure(int d0, int k, int nseed, const int *seeds);
int dnnf90_tabset_nderiv(int tid);
int dnnf90_net_create_ts(int tid, int nlayer, const int *dims);

int dnnf90_net_load(const char *filename);          /* nn_weight.dat format */
/* Load onto a named table set.  dnnf90_net_load uses whatever table set
   is current, which a weight file trained with a non-tanh activation is
   refused against; set the activation on a table set first and load
   through this entry instead.  Codes: 0 TANH, 1 SIN, 2 ERF, 3 BESSEL
   (J_0), 4 BESSEL1 (J_1). */
int dnnf90_tabset_activation(int tid, int iact);
int dnnf90_net_load_ts(int tid, const char *filename);
int dnnf90_net_create(int nlayer, const int *dims);
int dnnf90_net_set_layer(int nid, int l, const double *wrow);
int dnnf90_net_free(int nid);

/* evaluation (value and all carried mixed derivatives) */
int dnnf90_work_create(int nid);
int dnnf90_work_free(int wid);
int dnnf90_eval(int nid, int wid, const double *x, double *t);

/* training (gradient accumulation and Adam) */
int dnnf90_twork_create(int nid);
int dnnf90_twork_free(int tid);
int dnnf90_grad_create(int nid);
int dnnf90_grad_free(int gid);
int dnnf90_grad_zero(int gid);
int dnnf90_grad_point(int nid, int tid, const double *x,
                      const double *seed, int gid);
int dnnf90_adam_step(int nid, int gid, double eta, double beta1,
                     double beta2, double eps, int nbatch, int istep);

#ifdef __cplusplus
}
#endif
#endif
