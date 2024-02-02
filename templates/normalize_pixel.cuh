#ifndef NORMALIZE_PIXEL_CUH
#define NORMALIZE_PIXEL_CUH

#include "kernel.h"

// explicit instantiations
template __device__ __forceinline__ void normalize_pixel<3u>(Pixel<3u> *target, int pixel_idx, 
                                                    const Pixel<3u> *smallest, const Pixel<3u> *largest);
template __device__ __forceinline__ void normalize_pixel<4u>(Pixel<4u> *target, int pixel_idx,
                                                    const Pixel<4u> *smallest, const Pixel<4u> *largest);


#endif