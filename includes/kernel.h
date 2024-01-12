#ifndef __KERNELS__H
#define __KERNELS__H

#include <stdio.h>
#include <assert.h>

#include "pixel.h"

#define OUT_OF_BOUNDS -1

#define CUDA_CHECK_ERROR(errorMessage) do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        fprintf(stderr, "    %s\n", errorMessage); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct kernel_data {
    // shift values are expected to be [0, 100] (percentage)
    // i.e red_shift = 0 -> no red shift, red_shift = 100 -> red channel is 100% of the channel value
    bool                        normalize; // false means we clamp values to [0, 255] to be able to display them,
                                           // true means we also perform linear normalization
    unsigned char               red_shift;
    unsigned char               green_shift;
    unsigned char               blue_shift;
    unsigned char               alpha_shift; // 0 if no alpha channel, so just do nothing
    unsigned char               brightness; // 0 if no brightness change, 100 if we want to double the brightness
};

// Returns a 1D indexing of a 2D array index, returns -1 if out of bounds
__device__ __forceinline__ int32_t find_index(int width, int height, int row, int column) {
    if (row >= 0 && row < height && column >= 0 && column < width) {
        return row * width + column;
    }
    // -1 if out of bounds, returns 1D array indexing otherwise
    return OUT_OF_BOUNDS;
}

// smallest and largest hold the smallest and largest pixel values for each channel
template <unsigned int channels>
__device__ __forceinline__ void normalize_pixel(Pixel<channels> *target, int pixel_idx, 
                                                    const Pixel<channels> *smallest, const Pixel<channels> *largest) {
    // normalize each respective channel
    for (int channel = 0; channel < channels; channel++) {
        int min = smallest->data[channel];
        int max = largest->data[channel];
        int value = target[pixel_idx].data[channel];

        if(max == min) {
            max = (min + 1) % 255;
            min = (min - 1) % 255;
            if(min < 0) min += 255;
        } // to prevent division by zero (... / (max - min))
        // just assign random values if max = min 
        // this shouldnt happen in real life, so we'll just ignore it
        // although i am unsure if this is the best way to handle this

        target[pixel_idx].data[channel] = (value - min) * 255 / (max - min);
    }
}

// applies the filter to the input image at the given row and column
// returns sum of filter application
template<unsigned int channels>
__device__ __forceinline__ int apply_filter(const Pixel<channels> *input, const int8_t *filter, unsigned int mask,
    int dimension, int width, int height, int row, int col) {

    assert(mask <= channels);
    
    int sum = 0;
    int start_i = row - dimension / 2;
    int start_j = col - dimension / 2;

    // iterate over the filter
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            int filter_x = start_i + i;
            int filter_y = start_j + j;

            int filter_idx = find_index(width, height, filter_x, filter_y);

            if (filter_idx != OUT_OF_BOUNDS) {
                int member_value = input[filter_idx].data[mask];
                int8_t filter_value = filter[i * dimension + j];
                sum += member_value * filter_value;
            }
        }
    }
    return sum;
}

template <unsigned int channels>
void run_kernel(const int8_t *filter, int32_t dimension, const Pixel<channels> *input,
                 Pixel<channels> *output, int32_t width, int32_t height);

template <unsigned int channels>
__global__ void kernel(const int8_t *filter, int dimension,
                        const Pixel<channels> *input, Pixel<channels> *output, int width,
                        int height);

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *image, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *biggest);

// explicit instantiations
template __device__ __forceinline__ void normalize_pixel<3u>(Pixel<3u> *target, int pixel_idx, 
                                                    const Pixel<3u> *smallest, const Pixel<3u> *largest);
template __device__ __forceinline__ void normalize_pixel<4u>(Pixel<4u> *target, int pixel_idx,
                                                    const Pixel<4u> *smallest, const Pixel<4u> *largest);

template __device__ __forceinline__ int apply_filter<3u>(const Pixel<3u> *input, const int8_t *filter, unsigned int mask,
    int dimension, int width, int height, int row, int col);

template __device__ __forceinline__ int apply_filter<4u>(const Pixel<4u> *input, const int8_t *filter, unsigned int mask,
    int dimension, int width, int height, int row, int col);

#endif
