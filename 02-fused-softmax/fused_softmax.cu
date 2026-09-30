// Triton tutorial 02: Fused Softmax, in CUDA C++.
// https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
//
// Row-wise softmax of an M x N float32 matrix:
//   y[i, :] = exp(x[i, :] - max(x[i, :])) / sum(exp(x[i, :] - max(x[i, :])))
//
// The Triton kernel:
//
//   @triton.jit
//   def softmax_kernel(output_ptr, input_ptr, input_row_stride,
//   output_row_stride,
//                      n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages:
//                      tl.constexpr):
//       row_start = tl.program_id(0)
//       row_step = tl.num_programs(0)
//       for row_idx in tl.range(row_start, n_rows, row_step,
//       num_stages=num_stages):
//           row_start_ptr = input_ptr + row_idx * input_row_stride
//           col_offsets = tl.arange(0, BLOCK_SIZE)
//           mask = col_offsets < n_cols
//           row = tl.load(row_start_ptr + col_offsets, mask=mask,
//           other=-float('inf')) row_minus_max = row - tl.max(row, axis=0)
//           numerator = tl.exp(row_minus_max)
//           denominator = tl.sum(numerator, axis=0)
//           softmax_output = numerator / denominator
//           output_row_start_ptr = output_ptr + row_idx * output_row_stride
//           tl.store(output_row_start_ptr + col_offsets, softmax_output,
//           mask=mask)
//
// How the ideas map:
//   persistent program looping over rows  -> thread block with a grid-stride
//   loop over rows num_programs = NUM_SM * occupancy     ->
//   cudaOccupancyMaxActiveBlocksPerMultiprocessor num_warps = 8 -> 256 threads
//   per block BLOCK_SIZE = next_power_of_2(n_cols)  -> template parameter: each
//   thread keeps
//                                            BLOCK_SIZE / 256 values of the row
//                                            in registers
//   tl.max / tl.sum over the row          -> block_reduce() from
//   common/reduce.cuh tl.exp (fast but approximate)         -> __expf
//
// Triton's `num_stages` also pipelines the loop over rows: it loads the next
// row while the current one is being reduced. That is left out here;
// prefetching the next row into a second register array is a good exercise.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <vector>

#include "bench.cuh"
#include "cuda_utils.cuh"
#include "reduce.cuh"

constexpr int kNumThreads = 256;  // Triton: num_warps = 8
constexpr int kMaxBlockSize =
    16384;  // longest row kept in registers (64 floats per thread)

// ---------------------------------------------------------------------------
// Fused softmax: every row is read from DRAM once and written once.
// ---------------------------------------------------------------------------
template <int BLOCK_SIZE, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    softmax_kernel(float* __restrict__ output, const float* __restrict__ input,
                   int input_row_stride, int output_row_stride, int n_rows,
                   int n_cols) {
    // Thread t owns columns t, t + NUM_THREADS, t + 2 * NUM_THREADS, ... so
    // each load instruction of a warp reads 32 consecutive floats (coalesced).
    constexpr int ITEMS =
        BLOCK_SIZE > NUM_THREADS ? BLOCK_SIZE / NUM_THREADS : 1;
    __shared__ float scratch[NUM_THREADS / 32];

    for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float* in_row = input + int64_t(row) * input_row_stride;
        float x[ITEMS];  // the row, spread over the block's registers
#pragma unroll
        for (int i = 0; i < ITEMS; i++) {
            const int col = threadIdx.x + i * NUM_THREADS;
            x[i] =
                col < n_cols ? in_row[col] : -INFINITY;  // mask with other=-inf
        }

        float row_max = -INFINITY;
#pragma unroll
        for (int i = 0; i < ITEMS; i++) {
            row_max = fmaxf(row_max, x[i]);
        }
        row_max = block_reduce<NUM_THREADS>(row_max, MaxOp(), scratch);
        float denominator = 0.f;
#pragma unroll
        for (int i = 0; i < ITEMS; i++) {
            x[i] = __expf(x[i] - row_max);  // padding: exp(-inf) = 0
            denominator += x[i];
        }
        denominator = block_reduce<NUM_THREADS>(denominator, SumOp(), scratch);

        float* out_row = output + int64_t(row) * output_row_stride;
#pragma unroll
        for (int i = 0; i < ITEMS; i++) {
            const int col = threadIdx.x + i * NUM_THREADS;
            if (col < n_cols) {
                out_row[col] = x[i] / denominator;
            }
        }
    }
}

using SoftmaxKernel = void (*)(float*, const float*, int, int, int, int);

struct SoftmaxLaunch {
    SoftmaxKernel kernel;
    int blocks_per_sm;
};

// The counterpart of Triton's per-BLOCK_SIZE kernel cache: pick the template
// instance and compute its occupancy once.
const SoftmaxLaunch& softmax_launch_for(int block_size) {
    static std::map<int, SoftmaxLaunch> cache;
    auto it = cache.find(block_size);
    if (it != cache.end()) {
        return it->second;
    }

    SoftmaxKernel kernel = nullptr;
    switch (block_size) {
        case 256:
            kernel = softmax_kernel<256, kNumThreads>;
            break;
        case 512:
            kernel = softmax_kernel<512, kNumThreads>;
            break;
        case 1024:
            kernel = softmax_kernel<1024, kNumThreads>;
            break;
        case 2048:
            kernel = softmax_kernel<2048, kNumThreads>;
            break;
        case 4096:
            kernel = softmax_kernel<4096, kNumThreads>;
            break;
        case 8192:
            kernel = softmax_kernel<8192, kNumThreads>;
            break;
        case 16384:
            kernel = softmax_kernel<16384, kNumThreads>;
            break;
        default:
            std::fprintf(stderr, "softmax: unsupported BLOCK_SIZE %d\n",
                         block_size);
            std::exit(EXIT_FAILURE);
    }
    // Triton derives occupancy from the compiled kernel's register and shared
    // memory usage by hand; the CUDA runtime can do that for us.
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, kNumThreads, 0));
    return cache[block_size] =
               SoftmaxLaunch{kernel, std::max(blocks_per_sm, 1)};
}

// Host wrapper: the counterpart of the Python `softmax(x)`.
void softmax(float* y, const float* x, int n_rows, int n_cols) {
    if (n_rows == 0 || n_cols == 0) return;
    if (n_cols > kMaxBlockSize) {
        std::fprintf(stderr, "softmax: rows longer than %d are not supported\n",
                     kMaxBlockSize);
        std::exit(EXIT_FAILURE);
    }
    // Every block size <= 256 behaves the same (one value per thread).
    const int block_size = std::max(next_pow2(n_cols), kNumThreads);
    const SoftmaxLaunch& launch = softmax_launch_for(block_size);
    // Persistent grid: just enough blocks to fill the GPU once.
    const int num_programs = std::min(num_sms() * launch.blocks_per_sm, n_rows);
    launch.kernel<<<num_programs, kNumThreads>>>(y, x, n_cols, n_cols, n_rows,
                                                 n_cols);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Naive softmax: the tutorial's `naive_softmax`, one kernel per PyTorch op.
//   x_max = x.max(dim=1)                      read MN,      write M
//   z = x - x_max[:, None]                    read MN + M,  write MN
//   numerator = torch.exp(z)                  read MN,      write MN
//   denominator = numerator.sum(dim=1)        read MN,      write M
//   ret = numerator / denominator[:, None]    read MN + M,  write MN
// In total it reads 5MN + 2M elements and writes 3MN + 2M, against MN and MN
// for the fused kernel: up to ~4x more DRAM traffic.
// ---------------------------------------------------------------------------
constexpr int kNaiveThreads = 256;

// One block per row: out[row] = reduce(in[row, :]).
template <typename Op>
__global__ void row_reduce_kernel(float* __restrict__ out,
                                  const float* __restrict__ in, int n_cols,
                                  Op op) {
    __shared__ float scratch[kNaiveThreads / 32];
    const float* row = in + int64_t(blockIdx.x) * n_cols;
    float acc = Op::identity();
    for (int col = threadIdx.x; col < n_cols; col += kNaiveThreads) {
        acc = op(acc, row[col]);
    }
    acc = block_reduce<kNaiveThreads>(acc, op, scratch);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = acc;
    }
}

// In place: data[i] = exp(data[i]).
__global__ void exp_kernel(float* data, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * kNaiveThreads + threadIdx.x;
    if (i < n) data[i] = expf(data[i]);
}

// One block per row: out[row, :] = num[row, :] / v[row].
__global__ void div_rowwise_kernel(float* __restrict__ out,
                                   const float* __restrict__ num,
                                   const float* __restrict__ v, int n_cols) {
    const int64_t base = int64_t(blockIdx.x) * n_cols;
    const float s = v[blockIdx.x];
    for (int col = threadIdx.x; col < n_cols; col += kNaiveThreads)
        out[base + col] = num[base + col] / s;
}

// One block per row: z[row, :] = x[row, :] - v[row].
__global__ void sub_rowwise_kernel(float* __restrict__ z,
                                   const float* __restrict__ x,
                                   const float* __restrict__ v, int n_cols) {
    const int64_t base = int64_t(blockIdx.x) * n_cols;
    const float s = v[blockIdx.x];
    for (int col = threadIdx.x; col < n_cols; col += kNaiveThreads) {
        z[base + col] = x[base + col] - s;
    }
}

// `tmp` holds n_rows * n_cols floats; `row_buf` holds n_rows floats.
void naive_softmax(float* y, const float* x, float* tmp, float* row_buf,
                   int n_rows, int n_cols) {
    if (n_rows == 0 || n_cols == 0) return;
    const int64_t n = int64_t(n_rows) * n_cols;
    row_reduce_kernel<<<n_rows, kNaiveThreads>>>(row_buf, x, n_cols, MaxOp());
    sub_rowwise_kernel<<<n_rows, kNaiveThreads>>>(tmp, x, row_buf, n_cols);
    exp_kernel<<<unsigned(cdiv<int64_t>(n, kNaiveThreads)), kNaiveThreads>>>(
        tmp, n);
    row_reduce_kernel<<<n_rows, kNaiveThreads>>>(row_buf, tmp, n_cols, SumOp());
    div_rowwise_kernel<<<n_rows, kNaiveThreads>>>(y, tmp, row_buf, n_cols);
    CUDA_CHECK_LAST();
}

// ---------------------------------------------------------------------------
// Correctness test. The tutorial uses a 1823 x 781 matrix so that neither
// dimension is a power of two; the other shapes cover every BLOCK_SIZE.
// ---------------------------------------------------------------------------
void softmax_reference(float* y, const float* x, int n_rows, int n_cols) {
    for (int r = 0; r < n_rows; r++) {
        const float* xr = x + int64_t(r) * n_cols;
        double m = -INFINITY, s = 0.0;
        for (int c = 0; c < n_cols; c++) {
            m = std::max(m, double(xr[c]));
        }
        for (int c = 0; c < n_cols; c++) {
            s += std::exp(double(xr[c]) - m);
        }
        for (int c = 0; c < n_cols; c++) {
            y[int64_t(r) * n_cols + c] = float(std::exp(double(xr[c]) - m) / s);
        }
    }
}

// Same rule as torch.allclose (default rtol=1e-5, atol=1e-8).
bool allclose(const std::vector<float>& got, const std::vector<float>& ref,
              double* max_diff, double rtol = 1e-5, double atol = 1e-8) {
    bool ok = true;
    *max_diff = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs(double(got[i]) - double(ref[i]));
        *max_diff = std::isnan(d) ? INFINITY : std::max(*max_diff, d);
        if (!(d <= atol + rtol * std::fabs(double(ref[i])))) {
            ok = false;
        }
    }
    return ok;
}

bool run_tests() {
    const std::pair<int, int> shapes[] = {{1823, 781}, {1, 1},    {3, 17},
                                          {5, 256},    {7, 257},  {4, 1000},
                                          {2, 16384},  {9, 12345}};
    bool all_ok = true;
    for (const auto& [n_rows, n_cols] : shapes) {
        const int64_t n = int64_t(n_rows) * n_cols;
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        std::vector<float> hx(n), expected(n), got(n);
        for (auto& v : hx) {
            v = dist(rng);
        }
        softmax_reference(expected.data(), hx.data(), n_rows, n_cols);

        DeviceBuffer<float> dx(n), dy(n), dtmp(n), drow(n_rows);
        dx.copy_from_host(hx.data(), n);

        for (const char* name : {"fused", "naive"}) {
            CUDA_CHECK(cudaMemset(dy.get(), 0xFF, n * sizeof(float)));  // NaN
            if (std::strcmp(name, "fused") == 0) {
                softmax(dy.get(), dx.get(), n_rows, n_cols);
            } else {
                naive_softmax(dy.get(), dx.get(), dtmp.get(), drow.get(),
                              n_rows, n_cols);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            dy.copy_to_host(got.data(), n);
            double max_diff = 0.0;
            const bool ok = allclose(got, expected, &max_diff);
            all_ok &= ok;
            std::printf("  %5d x %-5d  %-5s  max |diff| = %.3g  %s\n", n_rows,
                        n_cols, name, max_diff, ok ? "OK" : "FAIL");
        }
    }
    return all_ok;
}

// ---------------------------------------------------------------------------
// Benchmark: M = 4096 rows, N = 256 .. 12672 columns. As in the tutorial,
// bandwidth is always computed from 2 * M * N * 4 bytes (one read, one write),
// so the naive version's lower number reflects its extra traffic.
// ---------------------------------------------------------------------------
void run_benchmark() {
    constexpr int M = 4096;
    constexpr int kMaxN = 128 * 99;
    const int64_t max_elems = int64_t(M) * kMaxN;
    DeviceBuffer<float> dx(max_elems), dy(max_elems), dtmp(max_elems), drow(M);
    {
        std::mt19937 rng(0);
        std::normal_distribution<float> dist(0.f, 1.f);
        std::vector<float> hx(max_elems);
        for (auto& v : hx) v = dist(rng);
        dx.copy_from_host(hx.data(), max_elems);
    }

    std::printf("\nsoftmax-performance (M=%d, GB/s, median [p20, p80]):\n", M);
    std::printf("%8s %28s %28s\n", "N", "fused", "naive");
    for (int N = 256; N <= kMaxN; N += 128) {
        const double bytes = 2.0 * M * N * sizeof(float);
        const BenchResult fused =
            benchmark([&] { softmax(dy.get(), dx.get(), M, N); });
        const BenchResult naive = benchmark([&] {
            naive_softmax(dy.get(), dx.get(), dtmp.get(), drow.get(), M, N);
        });
        std::printf("%8d", N);
        for (const BenchResult& r : {fused, naive}) {
            std::printf("  %8.1f [%7.1f, %7.1f]", gbps(bytes, r.median_ms),
                        gbps(bytes, r.p80_ms), gbps(bytes, r.p20_ms));
        }
        std::printf("\n");
    }
}

int main(int argc, char** argv) {
    const bool test_only = argc > 1 && std::strcmp(argv[1], "--test-only") == 0;
    if (!print_device_info()) return EXIT_FAILURE;

    std::printf("\nCorrectness (vs. CPU reference in double precision):\n");
    if (!run_tests()) {
        std::fprintf(stderr, "Correctness check FAILED\n");
        return EXIT_FAILURE;
    }
    if (!test_only) run_benchmark();
    return EXIT_SUCCESS;
}