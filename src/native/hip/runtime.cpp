#include "mokert/common.hpp"
#include "mokert/native.hpp"
#include <hip/hip_runtime.h>

#include <format>
#include <iostream>
#include <stacktrace>
#include <stdexcept>

#if __cplusplus >= 202302L
#define print_stacktrace(out) \
    out << "stacktrace:\n";   \
    out << std::stacktrace::current() << "\n";
#else
#define print_stacktrace(out)
#endif

namespace moke {
template <> void check_status(hipError_t status) {
    if (status == hipSuccess) { return; }
    auto errinfo = hipGetErrorString(status);
    auto errmsg = std::format("hip runtime error: {}", errinfo);

    std::cerr << errmsg << "\n";
    print_stacktrace(std::cerr);
    throw std::runtime_error(std::move(errmsg));
}

void sync_device() {check_status(hipDeviceSynchronize()); }
} // namespace moke
