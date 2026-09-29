#pragma once

// A small C++ counterpart of triton.testing.do_bench.

#include <algorithm>
#include <vector>

#include "cuda_utils.cuh"

struct BenchResult {
    float median_ms;
    float p20_ms;
    float p80_ms;
};

// Times `fn` (which must launch its GPU work on the default stream) with CUDA
// events. As do_bench does, it writes to a 256 MiB buffer before every run to
// flush the L2 cache, so small problems are not timed while still in cache.
template <typename Fn>
BenchResult benchmark(Fn&& fn, int warmup = 25, int rep = 100) {
    static DeviceBuffer<char> l2_flush(256u << 20);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) {
        fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(rep);
    for (int i = 0; i < rep; i++) {
        CUDA_CHECK(cudaMemsetAsync(l2_flush.get(), 0, l2_flush.size()));
        CUDA_CHECK(cudaEventRecord(start));
        fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    auto q = [&](double p) { return times[size_t(p * (rep - 1))]; };
    return {q(0.5), q(0.2), q(0.8)};
}

// Effective memory bandwidth in GB/s for `bytes` moved in `ms` milliseconds.
inline double gbps(double bytes, double ms) {
    return bytes * 1e-9 / (ms * 1e-3);
}