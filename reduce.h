#ifndef __REDUCE_H__
#define __REDUCE_H__

#include <kernel.h>

#define PIXEL_MAX false
#define PIXEL_MIN true


template<int channels>
__device__ __forceinline__ Pixel<channels> pixel_max(Pixel<channels> *a, Pixel<channels> *b) {
    Pixel<channels> result;
    #pragma unroll
    for (int i = 0; i < channels; i++) {
        result.data[i] = max(a.data[i], b.data[i]);
    }
    return result;
}

template<int channels>
__device__ __forceinline__ Pixel<channels> pixel_min(Pixel<channels> *a, Pixel<channels> *b) {
    Pixel<channels> result;
    #pragma unroll
    for (int i = 0; i < channels; i++) {
        result.data[i] = min(a.data[i], b.data[i]);
    }
}

#endif