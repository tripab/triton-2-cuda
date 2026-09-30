#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n  in `%s`\n",       \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Call after a kernel launch: catches bad launch configurations immediately.
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Ceiling division, same as triton.cdiv.
template <typename T>
__host__ __device__ constexpr T cdiv(T a, T b) {
    return (a + b - 1) / b;
}

// Smallest power of two >= n, same as triton.next_power_of_2.
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

// Number of SMs on the current device (queried once).
inline int num_sms() {
    static int count = [] {
        int dev = 0, n = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
        return n;
    }();
    return count;
}

// Prints the active device and returns false if no CUDA device is usable.
inline bool print_device_info() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "No CUDA-capable device found.\n");
        return false;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Device %d: %s (sm_%d%d, %d SMs, %.1f GiB)\n", dev, prop.name,
                prop.major, prop.minor, prop.multiProcessorCount,
                prop.totalGlobalMem / double(1 << 30));
    return true;
}

// RAII owner of a device allocation
template <typename T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t size() const { return count_; }

    void copy_from_host(const T* src, size_t count) {
        CUDA_CHECK(
            cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* dst, size_t count) {
        CUDA_CHECK(
            cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

   private:
    T* ptr_;
    size_t count_ = 0;
};