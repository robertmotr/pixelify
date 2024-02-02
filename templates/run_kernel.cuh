#ifndef PROCESS_IMAGE_CUH
#define PROCESS_IMAGE_CUH

#include "kernel.h"

// explicitly instantiate
template void run_kernel<3u>(const char* filter_name, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height, struct kernel_args extra);

template void run_kernel<4u>(const char* filter_name, const Pixel<4u> *input,
                  Pixel<4u> *output, int width, int height, struct kernel_args extra);

#endif