#pragma once
#include "mokert/native/runtime.hpp"
#include "mokert/native/memory.hpp"
#include "mokert/native/algorithm.hpp"

namespace moke {
template <typename T, auto CloseFunc>
MOKE_KERNEL void compare_all_close_kernel(int *equal, T const *lhs, T const *rhs, size_t length, float epsilon) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (; idx < length; idx += gridDim.x * blockDim.x) {
        if (!CloseFunc(lhs[idx], rhs[idx], epsilon)) {
            *equal = 0;
            return;
        }
    }
}

template <typename T, auto CloseFunc>
bool compare_all_close(const T *lhs, const T *rhs, size_t length, float epsilon) {
    int hres[1] = {1};
    auto dres = (int *)device_memory_t::malloc(sizeof(int));
    memory_copy<device_memory_t, host_memory_t>(dres, hres, sizeof(int));

    constexpr int nthreads = 1024;
    const int nblocks = (length + nthreads - 1) / nthreads;
    compare_all_close_kernel<T, CloseFunc><<<nblocks, nthreads>>>(dres, lhs, rhs, length, epsilon);
    sync_device();

    memory_copy<host_memory_t, device_memory_t>(hres, dres, sizeof(int));
    device_memory_t::free(dres);
    return *hres;
}

template <bool relative, class T>
bool compare_all_close(device_memory_t, const T *lhs, const T *rhs, size_t length, float epsilon) {
    constexpr auto compare_func = relative ? relative_close<T> : absolute_close<T>;
    return compare_all_close<T, compare_func>(lhs, rhs, length, epsilon);
}
} // namespace moke

#define DEVICE_COMPARE_ALL_CLOSE(T)                                                                 \
    template bool compare_all_close<true, T>(device_memory_t, const T *, const T *, size_t, float); \
    template bool compare_all_close<false, T>(device_memory_t, const T *, const T *, size_t, float);
