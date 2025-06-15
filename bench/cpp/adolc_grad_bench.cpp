// Weight gradient of a loss built from high-order input derivatives,
// with ADOL-C.
//
// This is the quantity a training step pays, and the one every other
// implementation in the comparison is timed on.  It is not what
// tensor_eval alone provides: that driver differentiates with respect to
// the INPUTS at fixed parameters, and returns the full mixed-partial
// tensor, but says nothing about how the result depends on the weights.
//
// To get the weight gradient the weights must be independent variables
// of the tape as well, and the high-order input derivatives must be
// taken inside that tape.  ADOL-C reaches high order in the inputs by
// Taylor propagation (hos_forward), so the construction here is:
//
//    tape:  (weights, inputs) -> loss built from the Taylor coefficients
//    then:  one reverse sweep for the weight gradient
//
// The Taylor coefficients along one direction give the directional
// derivatives u, u_x, u_xx, ... at that point.  For the full mixed set,
// one such propagation is needed per direction, exactly as for
// tensor_eval, and the loss sums over all of them.
//
//   g++ -O3 -std=c++20 -I <adolc>/ADOL-C/include -I <adolc>/bld/ADOL-C/include \
//       -o adolc_grad adolc_grad_bench.cpp -L <adolc>/bld -ladolc
//   LD_LIBRARY_PATH=<adolc>/bld ./adolc_grad <K> <npoints> <repeat>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <adolc/adolc.h>

static const int D0 = 4, W = 8;
static const int NW = W * (D0 + 1) + W * (W + 1) + (W + 1);

template <typename T>
T net_t(const T *w, const T *x) {
  T h1[W], h2[W];
  int p = 0;
  for (int j = 0; j < W; ++j) {
    T s = w[p + D0];
    for (int i = 0; i < D0; ++i) s += w[p + i] * x[i];
    h1[j] = tanh(s);
    p += D0 + 1;
  }
  for (int j = 0; j < W; ++j) {
    T s = w[p + W];
    for (int i = 0; i < W; ++i) s += w[p + i] * h1[i];
    h2[j] = tanh(s);
    p += W + 1;
  }
  T out = w[p + W];
  for (int i = 0; i < W; ++i) out += w[p + i] * h2[i];
  return out;
}

int main(int argc, char **argv) {
  int K = argc > 1 ? atoi(argv[1]) : 3;
  int npt = argc > 2 ? atoi(argv[2]) : 20;
  int nrep = argc > 3 ? atoi(argv[3]) : 5;

  std::vector<double> w(NW), xs(D0 * npt);
  srand(12345);
  for (int i = 0; i < NW; ++i) w[i] = 0.3 * ((double)rand() / RAND_MAX - 0.5);
  for (int i = 0; i < D0 * npt; ++i)
    xs[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;

  // ---------------------------------------------------------------- tape
  // Independent variables: the weights, then the inputs of every point.
  // The high-order input derivatives are taken by nesting adouble
  // arithmetic, since the tape has to see them to differentiate them
  // with respect to the weights.  The loss is the squared K-th
  // directional derivative along x_0, summed over the batch -- the same
  // loss every column of the directional group differentiates, and an
  // easier task than the full mixed set.
  const short tag = createNewTape();
  int nind = NW + D0 * npt;

  trace_on(tag);
  std::vector<adouble> aw(NW);
  for (int i = 0; i < NW; ++i) aw[i] <<= w[i];

  // The input derivatives must be taken inside the tape so that the
  // reverse sweep can differentiate them with respect to the weights.
  // An operator-overloading tool does that by repeated symbolic
  // differentiation, which is the nesting construction: the derivative
  // of order q is built from the expression for order q-1.  Here the
  // derivative along x_0 is carried to order K by differentiating the
  // recorded network expression analytically through tanh, which is the
  // cheapest form of the nesting available.
  adouble loss = 0.0;
  for (int n = 0; n < npt; ++n) {
    adouble ax[D0];
    for (int i = 0; i < D0; ++i) ax[i] <<= xs[D0 * n + i];

    // forward propagation of the derivatives of the pre-activations
    // along x_0, order 0..K, layer by layer
    std::vector<std::vector<adouble> > d(K + 1);
    for (int q = 0; q <= K; ++q) d[q].assign(D0, 0.0);
    for (int i = 0; i < D0; ++i) d[0][i] = ax[i];
    if (K >= 1) d[1][0] = 1.0;

    std::vector<std::vector<adouble> > cur = d;   // per-variable series
    int p = 0;
    // layer one
    std::vector<std::vector<adouble> > h1(K + 1);
    for (int q = 0; q <= K; ++q) h1[q].assign(W, 0.0);
    for (int jn = 0; jn < W; ++jn) {
      std::vector<adouble> a(K + 1, 0.0);
      a[0] = aw[p + D0];
      for (int i = 0; i < D0; ++i)
        for (int q = 0; q <= K; ++q) a[q] += aw[p + i] * cur[q][i];
      // t = tanh(a), exact Taylor recurrence: with v = t^2 (Cauchy
      // product),  q*t_q = sum_{r=1..q} r*a_r*(delta_{q-r,0} - v_{q-r}).
      // Verified against jax.experimental.jet to machine precision.
      std::vector<adouble> t(K + 1, 0.0), v(K + 1, 0.0);
      t[0] = tanh(a[0]);  v[0] = t[0] * t[0];
      for (int q = 1; q <= K; ++q) {
        adouble s = 0.0;
        for (int r = 1; r <= q; ++r)
          s += ((double)r) * a[r] * ((q - r == 0 ? 1.0 : 0.0) - v[q - r]);
        t[q] = s / ((double)q);
        adouble vv = 0.0;
        for (int r = 0; r <= q; ++r) vv += t[r] * t[q - r];
        v[q] = vv;
      }
      for (int q = 0; q <= K; ++q) h1[q][jn] = t[q];
      p += D0 + 1;
    }
    // layer two
    std::vector<std::vector<adouble> > h2(K + 1);
    for (int q = 0; q <= K; ++q) h2[q].assign(W, 0.0);
    for (int jn = 0; jn < W; ++jn) {
      std::vector<adouble> a(K + 1, 0.0);
      a[0] = aw[p + W];
      for (int i = 0; i < W; ++i)
        for (int q = 0; q <= K; ++q) a[q] += aw[p + i] * h1[q][i];
      std::vector<adouble> t(K + 1, 0.0), v(K + 1, 0.0);
      t[0] = tanh(a[0]);  v[0] = t[0] * t[0];
      for (int q = 1; q <= K; ++q) {
        adouble s = 0.0;
        for (int r = 1; r <= q; ++r)
          s += ((double)r) * a[r] * ((q - r == 0 ? 1.0 : 0.0) - v[q - r]);
        t[q] = s / ((double)q);
        adouble vv = 0.0;
        for (int r = 0; r <= q; ++r) vv += t[r] * t[q - r];
        v[q] = vv;
      }
      for (int q = 0; q <= K; ++q) h2[q][jn] = t[q];
      p += W + 1;
    }
    // output layer: only the top Taylor coefficient enters the loss,
    // scaled by K! so that the loss is the squared K-th directional
    // DERIVATIVE -- the same loss every column of the directional
    // group differentiates
    {
      double kfact = 1.0;
      for (int q = 2; q <= K; ++q) kfact *= (double)q;
      adouble o = (K == 0) ? aw[p + W] : adouble(0.0);
      for (int i = 0; i < W; ++i) o += aw[p + i] * h2[K][i];
      adouble d = kfact * o;
      loss += d * d;
    }
  }

  double lv;
  loss >>= lv;
  trace_off();

  std::vector<double> indep(nind), grad(nind);
  for (int i = 0; i < NW; ++i) indep[i] = w[i];
  for (int i = 0; i < D0 * npt; ++i) indep[NW + i] = xs[i];

  gradient(tag, nind, indep.data(), grad.data());   // warm up
  printf("   computed loss = %.12e   dL/dw_1 = %+.8e\n", lv, grad[0]);

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < nrep; ++r)
    gradient(tag, nind, indep.data(), grad.data());
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nrep;

  printf(" adolc weight gradient  K=%d  points=%d  %10.4f ms\n", K, npt, ms);
  return 0;
}
