// Cost of the full set of mixed partial derivatives with ADOL-C.
//
// ADOL-C is the closest published competitor to the method of this work,
// and for a reason worth stating: tensor_eval does NOT reach high order
// by nesting first-order differentiation.  It propagates truncated
// Taylor polynomials along a set of directions and recovers the mixed
// partials from them, which is the same idea as tabulating the
// high-order chain rule once and reusing it.  Every other implementation
// in this comparison nests.
//
// What tensor_eval returns is the full derivative tensor of a function
// of p variables to order d: all binomial(p+d, d) mixed partials.  That
// is exactly the set this library carries, so the two can be compared
// directly, unlike the nested C++ benchmarks which take one direction.
//
// The network is 4-8-8-1 with tanh, matching the hod setting elsewhere.
//
//   g++ -O3 -std=c++17 -I <adolc>/include -o adolc_bench adolc_bench.cpp \
//       -L <adolc>/lib -ladolc
//   LD_LIBRARY_PATH=<adolc>/lib ./adolc_bench <K> <npoints> <repeat>
//
// Note on what is timed.  tensor_eval differentiates with respect to the
// INPUTS at a fixed set of weights; the weight gradient of a loss built
// from those derivatives would need a further reverse sweep.  The
// forward part is what is compared here, and the library figure quoted
// beside it is its forward pass, so the two are the same quantity.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <adolc/adolc.h>

static const int D0 = 4, W = 8;
static const int NW = W * (D0 + 1) + W * (W + 1) + (W + 1);

template <typename T>
T net_t(const double *w, const T *x) {
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
  int nrep = argc > 3 ? atoi(argv[3]) : 10;

  std::vector<double> w(NW), xs(D0 * npt);
  srand(12345);
  for (int i = 0; i < NW; ++i) w[i] = 0.3 * ((double)rand() / RAND_MAX - 0.5);
  for (int i = 0; i < D0 * npt; ++i)
    xs[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;

  // ---- record the tape once: the network as a function of its inputs
  // recent ADOL-C allocates the tape identifier rather than taking one
  const short tag = createNewTape();
  trace_on(tag);
  adouble ax[D0], ay;
  for (int i = 0; i < D0; ++i) ax[i] <<= xs[i];
  ay = net_t<adouble>(w.data(), ax);
  double dummy;
  ay >>= dummy;
  trace_off();

  // ---- the derivative tensor, order K, over D0 directions
  int dim = binomi(D0 + K, K);          // number of mixed partials
  double **S = myalloc2(D0, D0);        // the identity: all coordinate dirs
  for (int i = 0; i < D0; ++i)
    for (int j = 0; j < D0; ++j) S[i][j] = (i == j) ? 1.0 : 0.0;
  double **tensor = myalloc2(1, dim);

  // warm up
  tensor_eval(tag, 1, D0, K, D0, xs.data(), tensor, S);

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < nrep; ++r)
    for (int n = 0; n < npt; ++n)
      tensor_eval(tag, 1, D0, K, D0, &xs[D0 * n], tensor, S);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nrep;

  printf(" adolc tensor_eval  K=%d  derivatives=%d  points=%d  %10.4f ms\n",
         K, dim, npt, ms);

  myfree2(S);
  myfree2(tensor);
  return 0;
}
