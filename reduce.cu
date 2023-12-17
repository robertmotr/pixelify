#include "reduce.h"

__global__ void reduce_max(int *in, int* out, unsigned int N)
{
    int max_val = INT_MIN; 

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        max_val = max(max_val, in[i]);
    }

    max_val = blockReduceMax(max_val);

    if (threadIdx.x == 0)
        atomicMax(out, max_val);
}

__global__ void reduce_min(int *in, int* out, unsigned int N)
{
    int min_val = INT_MAX;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        min_val = min(min_val, in[i]);
    }

    min_val = blockReduceMin(min_val);

    if (threadIdx.x == 0)
        atomicMin(out, min_val);
}