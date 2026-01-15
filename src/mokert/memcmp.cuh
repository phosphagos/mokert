#include "mokert/common.hpp"
#include "mokert/native/memory.hpp"
#include "mokert/native/runtime.hpp"

namespace moke {
MOKE_KERNEL void memcmp_kernel(int *equal, const char *lhs, const char *rhs, size_t length) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    for (; idx < length; idx += gridDim.x * blockDim.x) {
        if (lhs[idx] != rhs[idx]) {
            return (*equal = 0, void());
        }
    }
}

bool device_memory_t::memcmp(const void *lhs, const void *rhs, size_t length) {
    int hres[1] = {1};
    auto dres = (int *)device_memory_t::malloc(sizeof(int));
    memory_copy<device_memory_t, host_memory_t>(dres, hres, sizeof(int));

    constexpr int nthreads = 1024;
    const int nblocks = (length + nthreads - 1) / nthreads;
    memcmp_kernel<<<nblocks, nthreads>>>(dres, (const char *)lhs, (const char *)rhs, length);
    sync_device();

    memory_copy<host_memory_t, device_memory_t>(hres, dres, sizeof(int));
    device_memory_t::free(dres);
    return *hres;
}
} // namespace moke
