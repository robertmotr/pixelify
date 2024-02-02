#ifndef APPLY_FILTER_CUH
#define APPLY_FILTER_CUH

#include "kernel.h"

template __device__ __forceinline__ int apply_filter<3u>(const Pixel<3u> *input, const filter *filter, unsigned int mask,
    int width, int height, int row, int col);

template __device__ __forceinline__ int apply_filter<4u>(const Pixel<4u> *input, const filter *filter, unsigned int mask,
    int width, int height, int row, int col);

#endif