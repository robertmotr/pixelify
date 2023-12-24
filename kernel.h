#ifndef __KERNELS__H
#define __KERNELS__H

#include <stdio.h>

template <int channels>
struct Pixel {
    int data[channels];
};

#define CUDA_CHECK_ERROR(errorMessage) do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        fprintf(stderr, "    %s\n", errorMessage); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Returns a 1D indexing of a 2D array index, returns -1 if out of bounds
__device__ __forceinline__ int32_t find_index_cuda(int32_t width, int32_t height, int32_t row, int32_t column) {
    if (row >= 0 && row < height && column >= 0 && column < width) {
        return row * width + column;
    }
    // -1 if out of bounds, returns 1D array indexing otherwise
    return -1;
}

// smallest and largest hold the smallest and largest pixel values for each channel
template <int channels>
__device__ __forceinline__ void normalize_pixel_cuda(Pixel<channels> *target, int32_t pixel_idx, 
                                                    const Pixel<channels> *smallest, const Pixel<channels> *largest) {
    // normalize each respective channel
    for (int channel = 0; channel < channels; channel++) {
        int min = smallest[pixel_idx].data[channel];
        int max = largest[pixel_idx].data[channel];
        int value = target[pixel_idx].data[channel];

        if(max == min) continue; // to prevent division by zero (... / (max - min))
        target[pixel_idx].data[channel] = (value - min) * 255 / (max - min);
    }
}

// applies the filter to the input image at the given row and column
// returns sum of filter application
template <int channels>
__device__ __forceinline__ int apply_filter_cuda(const Pixel<channels> *input, const int8_t *filter, const unsigned char mask,
    int32_t dimension, int width, int height, int row, int col) {
    
    int32_t sum = 0;

    int start_i = row - dimension / 2;
    int start_j = col - dimension / 2;

    // Iterate over the filter
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            int filter_x = start_i + i;
            int filter_y = start_j + j;

            int filter_idx = find_index_cuda(width, height, filter_x, filter_y);

            if (filter_idx != -1) {
                int member_value = input[filter_idx].data[mask];
                int8_t filter_value = filter[i * dimension + j];
                sum += member_value * filter_value;
            }
        }
    }
    return sum;
}

template <int channels>
void run_kernel(const int8_t *filter, int32_t dimension, const Pixel<channels> *input,
                 Pixel<channels> *output, int32_t width, int32_t height);

template <int channels>
__global__ void kernel(const int8_t *filter, int32_t dimension,
                        const Pixel<channels> *input, Pixel<channels> *output, int32_t width,
                        int32_t height);

template<int channels>
__global__ void normalize(Pixel<channels> *image, int32_t width, int32_t height,
                           Pixel<channels> *smallest, Pixel<channels> *biggest);

#endif
