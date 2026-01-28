#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include "mokert/compare.cuh"
#include "mokert/random.cuh"
namespace moke {
template <class T>
void fill_random(device_memory_t, T *dest, size_t length, float min, float max, uint32_t seed, int bits) {
    return device_fill_random<hiprandState_t>(dest, length, min, max, seed, bits);
}

template <>
MOKERT_DEVICE void random_init(uint64_t seed, uint64_t subseq, uint64_t offset, hiprandState_t &state) {
    hiprand_init(seed, subseq, offset, &state);
}

template <>
MOKERT_DEVICE float random_uniform(hiprandState_t &state) { return hiprand_uniform(&state); }

template <class State>
MOKERT_DEVICE double random_uniform_double(State &state) { return hiprand_uniform_double(&state); }

DEVICE_FILL_RANDOM(float);
DEVICE_FILL_RANDOM(double);
DEVICE_FILL_RANDOM(__half);
DEVICE_FILL_RANDOM(__hip_bfloat16);

DEVICE_COMPARE_ALL_CLOSE(float);
DEVICE_COMPARE_ALL_CLOSE(double);
DEVICE_COMPARE_ALL_CLOSE(__half);
DEVICE_COMPARE_ALL_CLOSE(__hip_bfloat16);
} // namespace moke
