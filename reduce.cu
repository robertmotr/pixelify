#include "reduce.h"
#include "kernel.h"

template <unsigned int blockSize, unsigned int channels>
__device__ void warpReduce(volatile Pixel<channels> *sdata, unsigned int tid,
                           Pixel<channels> (*reduction_op)(Pixel<channels> *a, Pixel<channels> *b)) {
    if (blockSize >= 64) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 8]);
    if (blockSize >= 8) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 4]);
    if (blockSize >= 4) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 2]);
    if (blockSize >= 2) sdata[tid] = reduction_op(&sdata[tid], &sdata[tid + 1]);
}
template <unsigned int blockSize, int channels, bool reduction_type>
__global__ void reduce_pixels(Pixel<channels> *g_idata, Pixel<channels> *g_result, unsigned int n, bool reduction_op) {
    extern __shared__ Pixel<channels> sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    Pixel<channels> (*op)(Pixel<channels> *a, Pixel<channels> *b);
    op = (reduction_op == PIXEL_MAX) ? pixel_max : pixel_min;

    sdata[tid] = 0;
    while (i < n) {
        // Replace += with pixel-wise reduction using op()
        sdata[tid] = op(&sdata[tid], &g_idata[i]);
        sdata[tid] = op(&sdata[tid], &g_idata[i + blockSize]);
        i += gridSize;
    }
    __syncthreads();

        // double check if this if branch is actually needed
    // im doing it cuz the original code didnt have it back when GPU block sizes were 512 max
    if (blockSize >= 1024) {
        if (tid < 512) {
            sdata[tid] = op(&sdata[tid], &sdata[tid + 512]);
        }
        __syncthreads();
    } 

    if (blockSize >= 512) {
        if (tid < 256) {
            sdata[tid] = op(&sdata[tid], &sdata[tid + 256]);
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sdata[tid] = op(&sdata[tid], &sdata[tid + 128]);
        } 
        __syncthreads();
    }
    if (blockSize >= 128) {
         if (tid < 64) {
            sdata[tid] = op(&sdata[tid], &sdata[tid + 64]);
        } 
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_result[blockIdx.x] = sdata[0];
}
