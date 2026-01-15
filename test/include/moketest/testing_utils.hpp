#pragma once
#include "moketest/type_traits.hpp"
#include <mokert/native/memory.hpp>
#include <gtest/gtest.h>

#define ASSERT_IN(VALUE, LOWER, UPPER) \
    do {                               \
        ASSERT_GE(VALUE, LOWER);       \
        ASSERT_LE(VALUE, UPPER);       \
    } while (0)

namespace moke::test {
namespace traits {
    template <class T> struct GTestTypeList;

    template <template <class...> class T, class... Us>
    struct GTestTypeList<T<Us...>> : std::type_identity<::testing::Types<Us...>> {};
} // namespace traits

template <class T>
using GTestTypeList = typename traits::GTestTypeList<T>::type;

template <class T, class List>
using GTestCombination = GTestTypeList<Combine<T, List>>;

template <class... Lists>
using GTestProduction = GTestTypeList<Product<Lists...>>;
} // namespace moke::test

using memory_spaces = moke::TypeList<moke::host_memory_t, moke::device_memory_t>;

#if defined MOKE_PLATFORM_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using half_t = __half;
using bfloat16_t = __nv_bfloat16;
#elif defined MOKE_PLATFORM_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

using half_t = __half;
using bfloat16_t = __hip_bfloat16;
#endif // MOKE_PLATFORM
