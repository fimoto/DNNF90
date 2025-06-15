// This file (bench/cpp/xad_bench.cpp) is part of DNNF90.
// (MIT License; see LICENSE at the repository root.)
//
// Weight gradient of a directional-derivative loss with XAD.
//
// XAD, like CoDiPack, reaches high order by nesting its types: an
// adjoint tape whose scalar is a stack of forward duals.  The
// combinatorial growth in the order is therefore the same as for any
// nested first-order construction; what a C++ tool changes is the
// constant, since no interpreter is in the loop.
//
// The loss is  sum_n (d^K u/dx_1^K)^2  over the batch -- the K-th
// directional derivative along x_1, the same loss every column of the
// directional group differentiates.  Construction: active type
// AReal<F_K> with F_K a K-deep FReal stack; the K forward seeds carry
// the direction, the K-th derivative is the all-epsilon corner of the
// output payload, and one reverse sweep with the adjoint seeded at
// 2*(that corner) in its all-value corner returns dL/dw in the
// all-epsilon corner of the weight adjoints.  The computed loss and
// dL/dw_1 are printed so that a run is evidently a real computation;
// they can be checked against bench/post/refvals (see README).
//
//   g++ -O2 -std=c++17 -I xad/src -I xad/bld/src -o xad_bench \
//       xad_bench.cpp -L xad/bld/lib -lxad
//   ./xad_bench <K> <npoints> <repeat>
//
// Orders 1..7 are instantiated; the payload doubles per level, so
// compile time and memory grow with the maximum order built.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <XAD/XAD.hpp>

static const int D0 = 4, W = 8;
static const int NW = W * (D0 + 1) + W * (W + 1) + (W + 1);

template <typename T>
T net(const T *w, const T *x) {
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

// ---- accessors on a K-deep FReal stack ---------------------------------
template <typename T> struct fdepth { static constexpr int d = 0; };
template <typename U> struct fdepth<xad::FReal<U>> {
  static constexpr int d = fdepth<U>::d + 1;
};

inline double &valcorner(double &x) { return x; }
template <typename U> double &valcorner(xad::FReal<U> &x) {
  return valcorner(xad::value(x));
}
inline double &epscorner(double &x) { return x; }
template <typename U> double &epscorner(xad::FReal<U> &x) {
  return epscorner(xad::derivative(x));
}
// set the single-epsilon slot of level `lvl` (0 = innermost) to 1
template <typename U> void seed1(xad::FReal<U> &x, int lvl) {
  if (lvl == fdepth<xad::FReal<U>>::d - 1)
    valcorner(xad::derivative(x)) = 1.0;
  else if constexpr (fdepth<U>::d > 0)
    seed1(xad::value(x), lvl);
}

template <int K> struct FStack {
  using type = xad::FReal<typename FStack<K - 1>::type>;
};
template <> struct FStack<0> { using type = double; };

template <int K>
double run(const std::vector<double> &w, const std::vector<double> &xs,
           int npt, int nrep) {
  using F = typename FStack<K>::type;
  using Tape = xad::Tape<F>;
  using A = xad::AReal<F>;

  auto one = [&](bool report) {
    Tape tape;
    std::vector<A> wt(NW);
    for (int i = 0; i < NW; ++i) {
      wt[i] = w[i];
      tape.registerInput(wt[i]);
    }
    tape.newRecording();
    std::vector<A> ys(npt);
    double loss = 0.0;
    for (int n = 0; n < npt; ++n) {
      A xv[D0];
      for (int i = 0; i < D0; ++i) xv[i] = xs[D0 * n + i];
      for (int l = 0; l < K; ++l) seed1(xad::value(xv[0]), l);
      ys[n] = net<A>(wt.data(), xv);
      tape.registerOutput(ys[n]);
      double dn = epscorner(xad::value(ys[n]));   // K-th derivative
      loss += dn * dn;
      valcorner(xad::derivative(ys[n])) = 2.0 * dn;  // adjoint seed
    }
    tape.computeAdjoints();
    double g0 = epscorner(xad::derivative(wt[0]));
    if (report)
      printf("   computed loss = %.12e   dL/dw_1 = %+.8e\n", loss, g0);
    return g0;
  };

  one(true);  // warm up, and print the evidence line once
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < nrep; ++r) one(false);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / nrep;
}

int main(int argc, char **argv) {
  int order = argc > 1 ? atoi(argv[1]) : 1;
  int npt = argc > 2 ? atoi(argv[2]) : 20;
  int nrep = argc > 3 ? atoi(argv[3]) : 20;
  if (order < 1 || order > 7) {
    fprintf(stderr, "orders 1..7 are instantiated.\n");
    return 1;
  }

  std::vector<double> w(NW), xs(D0 * npt);
  srand(12345);
  for (int i = 0; i < NW; ++i) w[i] = 0.3 * ((double)rand() / RAND_MAX - 0.5);
  for (int i = 0; i < D0 * npt; ++i)
    xs[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;

  double ms = 0.0;
  switch (order) {
  case 1: ms = run<1>(w, xs, npt, nrep); break;
  case 2: ms = run<2>(w, xs, npt, nrep); break;
  case 3: ms = run<3>(w, xs, npt, nrep); break;
  case 4: ms = run<4>(w, xs, npt, nrep); break;
  case 5: ms = run<5>(w, xs, npt, nrep); break;
  case 6: ms = run<6>(w, xs, npt, nrep); break;
  case 7: ms = run<7>(w, xs, npt, nrep); break;
  }
  printf(" xad order=%d points=%d  %10.4f ms per gradient\n", order, npt, ms);
  return 0;
}
