#ifndef NORMALIZE_CUH
#define NORMALIZE_CUH

#include "kernel.h"

template void normalize(Pixel<3u> *target, int width, int height, const Pixel<3u> *smallest, 
               const Pixel<3u> *largest, bool normalize_or_clamp);

template void normalize(Pixel<4u> *target, int width, int height, const Pixel<4u> *smallest, 
               const Pixel<4u> *largest, bool normalize_or_clamp);

#endif