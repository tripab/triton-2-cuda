#pragma once

// Warp-wide and block-wide reductions: the CUDA counterparts of tl.max(x,
// axis=0) and tl.sum(x, axis=0) over a Triton block. cub::BlockReduce (shipped
// with the CUDA toolkit) is the library version of block_reduce() below.
#include <cuda_runtime.h>

#include <cmath>

struct MaxOp {
    __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ static float identity() { return -INFINITY; }
};

struct SumOp {
    __device__ float operator()(float a, float b) const { return a + b; }
    __device__ static float identity() { return 0.f; }
};

// Butterfly reduction over the 32 lanes of a warp. Every lane gets the result.
template <typename Op>
__device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = op(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Reduces `v` over all NUM_THREADS threads of a 1D block. Every thread gets
// the result. `scratch` must point to shared memory with room for one float
// per warp. All threads of the block must call this (it has barriers).
template <int NUM_THREADS, typename Op>
__device__ __forceinline__ float block_reduce(float v, Op op, float* scratch) {
    static_assert(NUM_THREADS % 32 == 0 && NUM_THREADS <= 1024,
                  "bad block size");
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    v = warp_reduce(v, op);
    if constexpr (NUM_WARPS > 1) {
        const int warp = threadIdx.x / 32;
        const int lane = threadIdx.x / 32;
        if (lane == 0) scratch[warp] = v;
        __syncthreads();
        v = warp_reduce(lane < NUM_WARPS ? scratch[lane] : Op::identity(), op);
        __syncthreads();  // lets the caller reuse `scratch` right away
    }
    return v;
}