// This file (bench/cpp/codipack_bench.cpp) is part of DNNF90.
// (MIT License; see LICENSE at the repository root.)
//
// Weight gradient of a directional-derivative loss with CoDiPack.
//
// CoDiPack reaches high order the way an operator-overloading tool does:
// by nesting its types, RealForwardGen<RealForwardGen<...>> over a
// reverse tape.  That is exactly the "nest first-order differentiation
// K times" construction, so the combinatorial growth in K is the same
// as for nested automatic differentiation in a Python framework; what a
// C++ tool changes is the constant, since no interpreter is in the loop.
//
// The loss is  sum_n (d^K u/dx_1^K)^2  over the batch -- the K-th
// directional derivative along x_1, the same loss every column of the
// directional group differentiates.  The computed loss and dL/dw_1 are
// printed so that a run is evidently a real computation; they can be
// checked against the reference values in the README.
//
// The order is fixed at compile time so that exactly one nesting depth
// is instantiated and the optimizer sees the same small function a
// hand-written single-order benchmark would give it:
//
//   g++ -O2 -DORDER=3 -I <codipack>/include -o codi_bench3 codipack_bench.cpp
//   ./codi_bench3 <npoints> <repeat>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "codi.hpp"

#ifndef ORDER
#define ORDER 2
#endif

static const int D0 = 4, W = 8;

template <typename T, typename P>
T net(const P *w, const T *x) {
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

static const int NW = W * (D0 + 1) + W * (W + 1) + (W + 1);

using Tape = codi::RealReverse;

template <int Q> struct Lvl {
  using T = codi::RealForwardGen<typename Lvl<Q - 1>::T>;
};
template <> struct Lvl<0> { using T = Tape; };
template <int Q> using L = typename Lvl<Q>::T;

// promote a tape scalar through Q forward levels
template <int Q> L<Q> promote(const Tape &v) {
  if constexpr (Q == 0) return v;
  else return L<Q>(promote<Q - 1>(v));
}
// the all-value corner of a Q-level stack, as an assignable tape scalar
template <int Q> Tape &valcorner(L<Q> &x) {
  if constexpr (Q == 0) return x;
  else return valcorner<Q - 1>(x.value());
}
// set the single-epsilon slot of forward level j (0-based) to 1
template <int Q> void seed1(L<Q> &x, int j) {
  if constexpr (Q > 0) {
    if (j == Q - 1) valcorner<Q - 1>(x.gradient()) = 1.0;
    else seed1<Q - 1>(x.value(), j);
  }
}
// the all-epsilon corner: gradient() taken Q times
template <int Q> Tape extract(const L<Q> &y) {
  if constexpr (Q == 0) return y;
  else return extract<Q - 1>(y.getGradient());
}

int main(int argc, char **argv) {
  int npt = argc > 1 ? atoi(argv[1]) : 20;
  int nrep = argc > 2 ? atoi(argv[2]) : 20;

  std::vector<double> w(NW), xs(D0 * npt);
  srand(12345);
  for (int i = 0; i < NW; ++i) w[i] = 0.3 * ((double)rand() / RAND_MAX - 0.5);
  for (int i = 0; i < D0 * npt; ++i)
    xs[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;

  auto run = [&](bool report) {
    Tape::Tape &tape = Tape::getTape();
    std::vector<Tape> wt(NW);
    for (int i = 0; i < NW; ++i) wt[i] = w[i];
    tape.setActive();
    for (int i = 0; i < NW; ++i) tape.registerInput(wt[i]);

    Tape loss = 0.0;
    for (int n = 0; n < npt; ++n) {
      std::vector<L<ORDER>> wl(NW);
      for (int i = 0; i < NW; ++i) wl[i] = promote<ORDER>(wt[i]);
      L<ORDER> xf[D0];
      for (int i = 0; i < D0; ++i) xf[i] = xs[D0 * n + i];
      for (int j = 0; j < ORDER; ++j) seed1<ORDER>(xf[0], j);
      L<ORDER> y = net<L<ORDER>, L<ORDER>>(wl.data(), xf);
      Tape d = extract<ORDER>(y);
      loss += d * d;
    }
    tape.registerOutput(loss);
    tape.setPassive();
    loss.setGradient(1.0);
    tape.evaluate();
    double g0 = codi::RealTraits::getPassiveValue(wt[0].getGradient());
    double lv = codi::RealTraits::getPassiveValue(loss.getValue());
    tape.reset();
    if (report)
      printf("   computed loss = %.12e   dL/dw_1 = %+.8e\n", lv, g0);
    return g0;
  };

  run(true);  // warm up, and print the evidence line once
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < nrep; ++r) run(false);
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / nrep;

  printf(" codipack order=%d points=%d  %10.4f ms per gradient\n", ORDER, npt,
         ms);
  return 0;
}
