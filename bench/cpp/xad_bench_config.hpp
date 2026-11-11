// This file (bench/cpp/xad_bench_config.hpp) is part of DNNF90.
// (MIT License; see LICENSE at the repository root.)
//
// Tape-container sizing for the nested XAD benchmark.
//
// XAD stores the tape's partial derivatives in chunks of a fixed number
// of *entries* (8 Mi by default), and allocates the first chunk when the
// tape is constructed.  An entry holds one value of the tape's scalar
// type, which for the K-deep FReal stack used here is 2^K doubles.  The
// first chunk is therefore 8 Mi x 520 B = 4.4 GB at K = 6 and
// 8 Mi x 1032 B = 8.7 GB at K = 7, requested before anything is recorded,
// although the whole tape of this benchmark holds about 6300 entries
// (6.5 MB at K = 7).  On a machine that refuses the reservation this
// surfaces as std::bad_alloc, not as a tape that does not fit.
//
// This header sizes the chunks of nested FReal tapes in bytes instead.
// It must be included before <XAD/XAD.hpp> in every translation unit
// that instantiates or uses these tapes (xad_bench.cpp and
// xad_tape_higher.cpp), so that all of them see the same tape layout.
// Single-level types keep XAD's defaults.

#pragma once

#include <XAD/TapeContainer.hpp>

#include <cstddef>
#include <utility>

#ifndef XAD_BENCH_CHUNK_BYTES
#define XAD_BENCH_CHUNK_BYTES (std::size_t(64) << 20)
#endif

namespace xad {

template <class Scalar, std::size_t N> struct FReal;

namespace bench_detail {
// A multiple of 128 entries, so that the chunk size in bytes is a
// multiple of the 128-byte alignment aligned_alloc is called with.
template <class E> constexpr std::size_t chunk_entries() {
  return XAD_BENCH_CHUNK_BYTES / sizeof(E) > 4096
             ? XAD_BENCH_CHUNK_BYTES / sizeof(E) / 128 * 128
             : 4096;
}
}  // namespace bench_detail

template <class U, std::size_t M, std::size_t N, class S>
struct TapeContainerTraits<FReal<FReal<U, M>, N>, S> {
  using T = FReal<FReal<U, M>, N>;
  using statements_type = ChunkContainer<std::pair<S, S>>;
#ifdef XAD_REDUCED_MEMORY
  using operations_type =
      OperationsContainer<T, S, bench_detail::chunk_entries<T>()>;
#else
  using operations_type = OperationsContainerPaired<
      T, S, bench_detail::chunk_entries<std::pair<T, S>>()>;
#endif
};

}  // namespace xad
