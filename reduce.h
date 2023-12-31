#ifndef __REDUCE_H__
#define __REDUCE_H__

#include "kernel.h"

#define PIXEL_MAX false
#define PIXEL_MIN true
#define WARP_SIZE 32

// use this instead of builtin min() because we use
// PIXEL_NULL_CHANNEL to indicate that a pixel is invalid
__device__ inline int find_min(int a, int b) {
    return (a < b && a != PIXEL_NULL_CHANNEL) ? a : b;
}

__device__ inline int warp_reduce(int val, bool op) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        if (op == PIXEL_MAX) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        } else {
            val = min(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }
    return val;
}

__device__ inline int block_reduce(int val, bool op) {
    static __shared__ int shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce(val, op);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    if(threadIdx.x < blockDim.x / WARP_SIZE) {
        val = shared[lane];
    } else {
        val = (op == PIXEL_MAX) ? INT_MIN : INT_MAX;
    }

    if(wid == 0) val = warp_reduce(val, op);

    return val;
}

template<unsigned int channels>
__global__ void reduce_image(const Pixel<channels> *pixels_in, Pixel<channels> *result, 
                            unsigned int n, bool op);
   
#endif