// This file (bench/cpp/xad_tape_higher.cpp) is part of DNNF90.
// (MIT License; see LICENSE at the repository root.)
//
// Tape instantiations for the higher-order XAD benchmark.  The XAD
// library ships tape instantiations up to second order only; the
// directional benchmark nests deeper.  This translation unit compiles
// XAD's own Tape.cpp (so the standard instantiations come from here --
// do NOT also link libxad) and adds the deeper stacks.
//
//   g++ -O2 -std=c++17 -I <xad>/src -I <xad>/bld/src \
//       -o xad_bench xad_bench.cpp xad_tape_higher.cpp

#include <XAD/XAD.hpp>
#include <Tape.cpp>

template <int K> struct FStackI {
  using type = xad::FReal<typename FStackI<K - 1>::type>;
};
template <> struct FStackI<0> { using type = double; };

namespace xad {
template class Tape<FStackI<2>::type>;
template class Tape<FStackI<3>::type>;
template class Tape<FStackI<4>::type>;
template class Tape<FStackI<5>::type>;
template class Tape<FStackI<6>::type>;
template class Tape<FStackI<7>::type>;
}
