#ifndef IMAGE_REDUCTION_CUH
#define IMAGE_REDUCTION_CUH

#include "reduce.h"

template void image_reduction(const Pixel<3u> *d_input, Pixel<3u> *d_result, unsigned int size, bool operation);

template void image_reduction(const Pixel<4u> *d_input, Pixel<4u> *d_result, unsigned int size, bool operation);

#endif