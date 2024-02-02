#ifndef OTHER_KERNEL_CUH
#define OTHER_KERNEL_CUH

#include "kernel.h"

template void other_kernel(const Pixel<3u> *in, Pixel<3u> *out, int width, int height,
                              unsigned char operation, struct kernel_args extra);


template void other_kernel(const Pixel<4u> *in, Pixel<4u> *out, int width, int height,
                              unsigned char operation, struct kernel_args extra);


#endif