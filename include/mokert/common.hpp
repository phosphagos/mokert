#pragma once
#include <cstdint>
#include <cstddef>
#include <type_traits>

#if defined __CUDACC__ || defined __HIPCC__
#define MOKERT_UNIFIED __host__ __device__
#define MOKERT_DEVICE __device__
#define MOKERT_HOST __host__
#define MOKERT_KERNEL __global__
#else
#define MOKERT_UNIFIED
#define MOKERT_HOST
#define MOKERT_DEVICE
#define MOKERT_KERNEL
#endif

#define MOKERT_INLINE MOKERT_UNIFIED inline
#define MOKERT_CONSTEXPR MOKERT_INLINE constexpr
#define MOKERT_CAPI extern "C"

namespace moke {
using size_t = std::size_t;
using index_t = std::int64_t;
} // namespace moke
