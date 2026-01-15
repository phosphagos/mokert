#include "mokert/mokert.hpp"
#include <gtest/gtest.h>

#if defined __clang__
#define OPTNONE [[clang::optnone]]
#elif defined __GNUC__
#define OPTNONE [[gnu::optimize("O0")]]
#else
#define OPTNONE
#endif

using namespace moke;
constexpr size_t length = 1 << 20;

TEST(TestProfiler, TestHostProfiler) {
    host_profiler profiler{};
    auto vec0 = host_vector<float>(length) | fill::random{};
    auto vec1 = host_vector<float>(length);

    for (auto _ : profiler) { vec1 = vec0; }
    auto time = profiler.elapsed(std::micro{});
    auto bw = profiler.perf(length * 2 * sizeof(float), std::giga{});
    auto bw_ref = length * 2 * sizeof(float) / time / 1000;
    EXPECT_FLOAT_EQ(bw, bw_ref);

    printf("host_timer: copies %zd fp32 in %.0lfus\n", length, time);
    printf("bandwidth: %lgGB/s\n", bw);
    EXPECT_GT(time, 0);
}

TEST(TestProfiler, TestDeviceProfiler) {
    device_profiler profiler{};
    auto vec0 = device_vector<float>(length) | fill::random{};
    auto vec1 = device_vector<float>(length);

    for (auto _ : profiler) { vec1 = vec0; }
    auto time = profiler.elapsed(std::micro{});
    auto bw = profiler.perf(length * 2 * sizeof(float), std::giga{});
    auto bw_ref = length * 2 * sizeof(float) / time / 1000;
    EXPECT_FLOAT_EQ(bw, bw_ref);

    printf("device_timer: copies %zd fp32 in %lgus\n", length, time);
    printf("bandwidth: %lgGB/s\n", bw);
    EXPECT_GT(time, 0);
}
