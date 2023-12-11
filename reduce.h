#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <stdio.h>
#define warpSize 32

__device__ inline int warpReduceMax(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ inline int warpReduceMin(int val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ inline int blockReduceMax(int val) {
    static __shared__ int shared[32]; // Shared mem for 32 partial max values
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMax(val); // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : INT_MIN;

    if (wid == 0) val = warpReduceMax(val); // Final reduce within the first warp

    return val;
}

__device__ inline int blockReduceMin(int val) {
    static __shared__ int shared[32]; // Shared mem for 32 partial min values
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceMin(val); // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : INT_MAX;

    if (wid == 0) val = warpReduceMin(val); // Final reduce within the first warp

    return val;
}

__global__ void reduce10_max(int32_t *in, int32_t *out, unsigned int n);
__global__ void reduce10_min(int32_t *in, int32_t *out, unsigned int n);

#endif