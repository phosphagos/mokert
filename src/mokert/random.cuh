#include "mokert/native.hpp"

namespace moke {
template <class State>
MOKERT_DEVICE void random_init(uint64_t seed, uint64_t subseq, uint64_t offset, State &state);

template <class State>
MOKERT_DEVICE float random_uniform(State &state);

template <class State>
MOKERT_DEVICE double random_uniform_double(State &state);

template <class T, class State>
MOKERT_DEVICE T random_uniform(State &state, float alpha, float beta) {
    float res = std::is_same_v<T, double> ? random_uniform_double(state)
                                          : random_uniform(state);
    return T(res * alpha + beta);
}

template <class State, class T>
MOKERT_KERNEL void fill_random_kernel(T *dest, size_t length, float min, float max, uint32_t seed) {
    State state;

    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto tno = blockDim.x * gridDim.x;
    float alpha = max - min;
    float beta = min;

    if (tid < length) { random_init(seed, tid, 0, state); }
    for (auto i = tid; i < length; i += tno) {
        dest[tid] = random_uniform<T>(state, alpha, beta);
    }
}

template <class State, class T>
void device_fill_random(T *dest, size_t length, float min, float max, uint32_t seed) {
    constexpr int nthreads = 1024;
    constexpr int nblocks = 256;
    fill_random_kernel<State><<<nthreads, nblocks>>>(dest, length, min, max, seed);
}
} // namespace moke

#define DEVICE_FILL_RANDOM(T) \
    template void fill_random(device_memory_t, T *, size_t, float, float, uint32_t);
