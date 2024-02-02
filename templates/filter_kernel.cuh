#ifndef FILTER_KERNEL_CUH
#define FILTER_KERNEL_CUH

#include "kernel.h"

template void filter_kernel(const Pixel<3u> *in, Pixel<3u> *out, int width, int height,
                              const filter *filter, const struct kernel_args args);


template void filter_kernel(const Pixel<4u> *in, Pixel<4u> *out, int width, int height,
                              const filter *filter, const struct kernel_args args);

#endif 