// Triton tutorial 01: Vector Addition, in CUDA C++.
// https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
//
// The Triton kernel:
//
//   @triton.jit
//   def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:
//   tl.constexpr):
//       pid = tl.program_id(axis=0)
//       block_start = pid * BLOCK_SIZE
//       offsets = block_start + tl.arange(0, BLOCK_SIZE)
//       mask = offsets < n_elements
//       x = tl.load(x_ptr + offsets, mask=mask)
//       y = tl.load(y_ptr + offsets, mask=mask)
//       output = x + y
//       tl.store(output_ptr + offsets, output, mask=mask)
//
// How the ideas map:
//   Triton "program" (one instance)    -> CUDA thread block
//   tl.program_id(0)                   -> blockIdx.x
//   tl.arange(0, BLOCK_SIZE)           -> split across the block's threads by
//   hand mask=offsets < n                   -> `if (offset < n)` num_warps
//   (defaults to 4)          -> blockDim.x = 4 * 32 = 128 threads grid =
//   (cdiv(n, BLOCK_SIZE),)      -> gridDim.x = cdiv(n, BLOCK_SIZE)

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "bench.cuh"
#include "cuda_utils.cuh"

const int kBlockSize = 1024;  // elements per program/thread block
const int kNumThreads = 128;  // Triton's default num_warps=4

// ---------------------------------------------------------------------------
// Kernel 1: direct translation.
// Each block handles BLOCK_SIZE contiguous elements. On each loop step,
// consecutive threads touch consecutive addresses, so the loads and stores
// are coalesced.
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    add_kernel(const float* __restrict__ x, const float* __restrict__ y,
               float* __restrict__ out, int64_t n) {
    static_assert(BLOCK_SIZE % NUM_THREADS == 0,
                  "BLOCK_SIZE must be a multiple of NUM_THREADS");
    const int64_t block_start = int64_t(blockIdx.x) * BLOCK_SIZE;

#pragma unroll
    for (int i = threadIdx.x; i < BLOCK_SIZE; i += NUM_THREADS) {
        const int64_t offset = block_start + i;
        if (offset < n) {  // mask
            out[offset] = x[offset] + y[offset];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: vectorized. This is closer to the code Triton actually generates:
// the compiler gives each thread several contiguous elements and uses 128-bit
// loads and stores (ld.global.v4.f32). Full tiles need no masking. Only the
// last block (the tail) falls back to masked scalar code.
// Requires x, y, and out to be 16-byte aligned (cudaMalloc guarantees this).
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    add_kernel_vec4(const float* __restrict__ x, const float* __restrict__ y,
                    float* __restrict__ out, int64_t n) {
    constexpr int VEC = 4;
    static_assert(BLOCK_SIZE % (NUM_THREADS * VEC) == 0,
                  "BLOCK_SIZE must be a multiple of NUM_THREADS*4");
    const int64_t block_start = int64_t(blockIdx.x) * BLOCK_SIZE;

    if (block_start + BLOCK_SIZE <= n) {
        // Full tile: no mask needed
        const float4* x4 = reinterpret_cast<const float4*>(x + block_start);
        const float4* y4 = reinterpret_cast<const float4*>(y + block_start);
        float4* out4 = reinterpret_cast<float4*>(out + block_start);
#pragma unroll
        for (int i = threadIdx.x; i < BLOCK_SIZE / VEC; i += NUM_THREADS) {
            const float4 a = x4[i];
            const float4 b = y4[i];
            out4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
    } else {
        // Tail tile: masked scalar path
        for (int i = threadIdx.x; i < BLOCK_SIZE; i += NUM_THREADS) {
            const int64_t offset = block_start + i;
            if (offset < n) {
                out[offset] = x[offset] + y[offset];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers: the counterpart of the Python `add(x, y)` wrapper.
// ---------------------------------------------------------------------------
void add(const float* x, const float* y, float* out, int64_t n) {
    if (n == 0) return;
    const unsigned grid = unsigned(cdiv<int64_t>(n, kBlockSize));
    add_kernel<kBlockSize, kNumThreads><<<grid, kNumThreads>>>(x, y, out, n);
    CUDA_CHECK_LAST();
}

void add_vec4(const float* x, const float* y, float* out, int64_t n) {
    if (n == 0) return;
    const unsigned grid = unsigned(cdiv<int64_t>(n, kBlockSize));
    add_kernel_vec4<kBlockSize, kNumThreads>
        <<<grid, kNumThreads>>>(x, y, out, n);
    CUDA_CHECK_LAST();
}

// Library baseline, the counterpart of `x + y` in PyTorch.
void add_thrust(const float* x, const float* y, float* out, int64_t n) {
    thrust::transform(thrust::device, x, x + n, y, out, thrust::plus<float>());
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Correctness test (the tutorial uses size = 98432 with torch.rand inputs).
// ---------------------------------------------------------------------------
using AddFn = void (*)(const float*, const float*, float*, int64_t);

bool run_tests() {
    bool ok = true;
    // 98432 is the tutorial's size. The others check edge cases: empty input,
    // inputs smaller than a block, and inputs one element off a block boundary.
    for (int64_t n : {int64_t(98432), int64_t(0), int64_t(1), int64_t(1023),
                      int64_t(1025), int64_t(4096)}) {
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        std::vector<float> hx(n), hy(n), expected(n), got(n);
        for (int64_t i = 0; i < n; ++i) {
            hx[i] = dist(rng);
            hy[i] = dist(rng);
            expected[i] = hx[i] + hy[i];
        }

        DeviceBuffer<float> dx(n + 1), dy(n + 1),
            dout(n + 1);  // +1: avoid 0-byte allocs
        dx.copy_from_host(hx.data(), n);
        dy.copy_from_host(hy.data(), n);

        const struct {
            const char* name;
            AddFn fn;
        } impls[] = {{"add_kernel", add},
                     {"add_kernel_vec4", add_vec4},
                     {"thrust", add_thrust}};
        for (const auto& impl : impls) {
            // Fill the output with NaN so any element the kernel skips shows
            // up.
            CUDA_CHECK(cudaMemset(dout.get(), 0xFF, (n + 1) * sizeof(float)));
            impl.fn(dx.get(), dy.get(), dout.get(), n);
            CUDA_CHECK(cudaDeviceSynchronize());
            dout.copy_to_host(got.data(), n);

            float max_diff = 0.f;
            for (int64_t i = 0; i < n; ++i) {
                const float d = std::fabs(got[i] - expected[i]);
                max_diff = std::isnan(d) ? INFINITY : std::fmax(max_diff, d);
            }
            const bool pass = max_diff == 0.f;
            ok &= pass;
            std::printf("  n=%-7lld %-16s max |diff| vs CPU = %g  %s\n",
                        (long long)n, impl.name, max_diff,
                        pass ? "OK" : "FAIL");
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Benchmark: sizes 2^12 .. 2^27, reporting GB/s (3 * n * 4 bytes moved).
// ---------------------------------------------------------------------------
void run_benchmark() {
    constexpr int64_t kMaxN = int64_t(1) << 27;
    DeviceBuffer<float> dx(kMaxN), dy(kMaxN), dout(kMaxN);
    CUDA_CHECK(cudaMemset(dx.get(), 0, kMaxN * sizeof(float)));
    CUDA_CHECK(cudaMemset(dy.get(), 0, kMaxN * sizeof(float)));

    std::printf("\nvector-add-performance (GB/s, median [p20, p80]):\n");
    std::printf("%12s %28s %28s %28s\n", "size", "thrust", "add_kernel",
                "add_kernel_vec4");
    for (int e = 12; e <= 27; ++e) {
        const int64_t n = int64_t(1) << e;
        const double bytes = 3.0 * n * sizeof(float);
        std::printf("%12lld", (long long)n);
        for (AddFn fn : {add_thrust, add, add_vec4}) {
            const BenchResult r =
                benchmark([&] { fn(dx.get(), dy.get(), dout.get(), n); });
            // Lower time means higher bandwidth, so p80 time gives the low end.
            std::printf("  %8.1f [%7.1f, %7.1f]", gbps(bytes, r.median_ms),
                        gbps(bytes, r.p80_ms), gbps(bytes, r.p20_ms));
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    const bool test_only = argc > 1 && std::strcmp(argv[1], "--test-only") == 0;
    if (!print_device_info()) return EXIT_FAILURE;

    std::printf("\nCorrectness:\n");
    if (!run_tests()) {
        std::fprintf(stderr, "Correctness check FAILED\n");
        return EXIT_FAILURE;
    }
    if (!test_only) run_benchmark();
    return EXIT_SUCCESS;
}