#ifndef __KERNELS__H
#define __KERNELS__H

#include <stdio.h>
#include <assert.h>
#include "filters.h"
#include "filter_impl.h"
#include "pixel.h"

#define OUT_OF_BOUNDS       -1
#define OP_SHIFT_COLOURS    0
#define OP_BRIGHTNESS       1
#define OP_TINT             2

#define CUDA_CHECK_ERROR(errorMessage) do { \
    cudaError_t error = cudaGetLastError(); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        fprintf(stderr, "    %s\n", errorMessage); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct kernel_args {
    bool                        normalize; // false means we clamp values to [0, 255] to be able to display them,
                                           // true means perform linear normalization instead
    
    // values below are expected to be in [-100, 100] range
    // 0 means do nothing, 0 < x < 100 means increase values by x%, 0 > x > -100 means decrease values by x%
    unsigned char               filter_strength; // how much of the filter to apply [0, 100]
    unsigned char               dimension;
    char                        red_shift;       
    char                        green_shift; 
    char                        blue_shift; 
    char                        alpha_shift; 
    char                        brightness;
    // chosen by colour picker
    unsigned char tint[4] =     {0, 0, 0, 0}; // [red, green, blue, alpha]
    float                       blend_factor;
    unsigned char               passes;
};

// Returns a 1D indexing of a 2D array index, returns -1 if out of bounds
__device__ __forceinline__ int find_index(int width, int height, int row, int column) {
    if (row >= 0 && row < height && column >= 0 && column < width) {
        return row * width + column;
    }
    // -1 if out of bounds, returns 1D array indexing otherwise
    return OUT_OF_BOUNDS;
}

// Clamps pixels to [0, 255] range in order to be represented in a png file
template<unsigned int channels>
__device__ __forceinline__ void clamp_pixels(Pixel<channels> *target, int pixel_idx) {
    for (int channel = 0; channel < channels; channel++) {
        if (target[pixel_idx].data[channel] < 0) {
            target[pixel_idx].data[channel] = 0;
        } else if (target[pixel_idx].data[channel] > 255) {
            target[pixel_idx].data[channel] = 255;
        }
    }
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
__device__ __forceinline__ int apply_filter(cudaTextureObject_t tex_obj, const filter *filter, unsigned int mask, int width, 
                                            int height, int row, int col) {
    assert(mask < channels);

    extern __shared__ float smem[];
    
    int sum = 0;
    int start_i = row - filter->filter_dimension / 2;
    int start_j = col - filter->filter_dimension / 2;

    // iterate over the filter
    #pragma unroll
    for (int i = 0; i < filter->filter_dimension; i++) {
        #pragma unroll
        for (int j = 0; j < filter->filter_dimension; j++) {

            int filter_x = start_i + i;
            int filter_y = start_j + j;

            if (find_index(width, height, filter_x, filter_y) != OUT_OF_BOUNDS) {
                int member_value = tex2D<Pixel<channels>>(tex_obj, (float) filter_x, (float) filter_y).data[mask];
                int filter_value = smem[i * filter->filter_dimension + j];
                sum += member_value * filter_value;
            }
        }
    }
    return sum;
}

// shifts the colour of the given channel by the given percentage specified in extra
// for example extra.red_shift = 50 means we increase the red channel by 50%
// channel_value is the original value of the channel we are shifting
__device__ __forceinline__ int shift_colours(int channel_value, struct kernel_args extra,
                                            unsigned int channel) {
    if(channel == 0) {
        return channel_value * (100 + extra.red_shift) / 100;
    }
    else if(channel == 1) {
        return channel_value * (100 + extra.green_shift) / 100;
    }
    else if(channel == 2) {
        return channel_value * (100 + extra.blue_shift) / 100;
    }
    else if(channel == 3) {
        return channel_value * (100 + extra.alpha_shift) / 100;
    }
}

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                   Pixel<channels>* output, int width, int height, struct kernel_args extra);

template<unsigned int channels>
__global__ void filter_kernel(const cudaTextureObject_t tex_obj, Pixel<channels> *out, int width, int height,
                              const filter *filter, const struct kernel_args args);

template<unsigned int channels>
__global__  void other_kernel(const Pixel<channels> *in, Pixel<channels> *out, int width, int height,
                              unsigned char operation, struct kernel_args extra);

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *image, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *biggest,
                           bool normalize_or_clamp);

// EXPLICIT INSTANTIATIONS
template __device__ __forceinline__ int apply_filter<3u>(cudaTextureObject_t tex_obj, const filter *filter,
                                                        unsigned int mask, int width, int height, int row, int col);

template __device__ __forceinline__ int apply_filter<4u>(cudaTextureObject_t tex_obj, const filter *filter,
                                                        unsigned int mask, int width, int height, int row, int col);

template __device__ __forceinline__ void normalize_pixel<3u>(Pixel<3u> *target, int pixel_idx, 
                                                    const Pixel<3u> *smallest, const Pixel<3u> *largest);
template __device__ __forceinline__ void normalize_pixel<4u>(Pixel<4u> *target, int pixel_idx,
                                                    const Pixel<4u> *smallest, const Pixel<4u> *largest);

template __device__ __forceinline__ void clamp_pixels<3u>(Pixel<3u> *target, int pixel_idx);

template __device__ __forceinline__ void clamp_pixels<4u>(Pixel<4u> *target, int pixel_idx);

#endif
