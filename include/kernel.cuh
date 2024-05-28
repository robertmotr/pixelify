#ifndef __KERNELS__H
#define __KERNELS__H

#include <stdio.h>
#include <assert.h>
#include "filters.h"
#include "filter_impl.h"
#include "pixel.h"

#define MAX_REDUCE              true
#define MIN_REDUCE              false

#define SHORT_MIN               -32768
#define SHORT_MAX               32767

#define MAX_FILTER_DIMENSION    15
#define MAX_FILTER_1D_SIZE      (MAX_FILTER_DIMENSION * MAX_FILTER_DIMENSION)

#define WARP_SIZE               32 // 32 threads per warp on most modern GPUs   

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
__forceinline__ __device__ __host__
int find_index(int width, int height, int row, int column) {
    if (row >= 0 && row < height && column >= 0 && column < width) {
        return row * width + column;
    }
    // -1 if out of bounds, returns 1D array indexing otherwise
    return OUT_OF_BOUNDS;
}

// Clamps pixels to [0, 255] range in order to be represented in a png file
template<unsigned int channels>
__host__ __device__ __forceinline__ void clamp_pixels(Pixel<channels> *target, int pixel_idx) {
    for (int channel = 0; channel < channels; channel++) {
        if (target[pixel_idx].at(channel) < 0) {
            target[pixel_idx].set(channel, 0);
        } else if (target[pixel_idx].at(channel) > 255) {
            target[pixel_idx].set(channel, 255);
        }
    }
}

// smallest and largest hold the smallest and largest pixel values for each channel
template <unsigned int channels>
__host__ __device__ __forceinline__ void  normalize_pixel(Pixel<channels> *target, int pixel_idx, 
                                                         const Pixel<channels> *smallest, const Pixel<channels> *largest) {
    // normalize each respective channel
    for (int channel = 0; channel < channels; channel++) {
        short min = smallest->at(channel);
        short max = largest->at(channel);
        short value = target[pixel_idx].at(channel);

        if(max == min) {
            #ifdef _DEBUG
                printf("Triggered max == min in normalize_pixel\n");
                printf("Max: %d, Min: %d\n", max, min);
            #endif

            max = (min + 1) % 255;
            min = (min - 1) % 255;
            if(min < 0) min += 255;
        } // to prevent division by zero (... / (max - min))
        // just assign random values if max = min 
        // this shouldnt happen in real life, so we'll just ignore it
        // although i am unsure if this is the best way to handle this

        target[pixel_idx].set(channel, (short)(((value - min) * 255) / (max - min)));

    }
}

// shifts the colour of the given channel by the given percentage specified in extra
// for example extra.red_shift = 50 means we increase the red channel by 50%
// channel_value is the original value of the channel we are shifting
__host__ __device__ __forceinline__ short shift_colours(int channel_value, struct filter_args extra,
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

    return SHORT_MIN; // on error
}

template<unsigned int channels>
__device__ __forceinline__ short apply_filter(const Pixel<channels> *in, unsigned int mask, int width, 
                                            int height, int row, int col);

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                   Pixel<channels>* output, int width, int height, struct filter_args
                 extra);

template<unsigned int channels>
__global__ void filter_kernel(const Pixel<channels> *in, Pixel<channels> *out, int width, int height,
                              const struct filter_args args);

template<unsigned int channels>
__global__ void shift_kernel(Pixel<channels> *d_pixels, int width, int height, const struct filter_args extra);

template<unsigned int channels>
__global__ void brightness_kernel(Pixel<channels> *d_pixels, int width, int height, const struct filter_args extra);

template<unsigned int channels>
__global__ void tint_kernel(Pixel<channels> *d_pixels, int width, int height, const struct filter_args extra);
                        
template<unsigned int channels>
__global__ void invert_kernel(Pixel<channels> *d_pixels, int width, int height, const struct filter_args extra);

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *image, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *biggest,
                           bool normalize_or_clamp);

template<unsigned int channels> __forceinline__ __device__ 
Pixel<channels> warp_reduce_pixels(Pixel<channels> pixel, bool reduce_type);

template<unsigned int channels> __forceinline__ __device__
Pixel<channels> block_reduce_pixels(Pixel<channels> pixel, bool reduce_type);

template<unsigned int channels>
void image_reduction(const Pixel<channels> *d_image, Pixel<channels> *d_result, int pixels, 
                               bool reduce_type, int block_size); 

// EXPLICIT INSTANTIATIONS    
template __device__ __forceinline__ void normalize_pixel<3u>(Pixel<3u> *target, int pixel_idx, 
                                                    const Pixel<3u> *smallest, const Pixel<3u> *largest);
template __device__ __forceinline__ void normalize_pixel<4u>(Pixel<4u> *target, int pixel_idx,
                                                    const Pixel<4u> *smallest, const Pixel<4u> *largest);

template __device__ __forceinline__ void clamp_pixels<3u>(Pixel<3u> *target, int pixel_idx);

template __device__ __forceinline__ void clamp_pixels<4u>(Pixel<4u> *target, int pixel_idx);

#endif
