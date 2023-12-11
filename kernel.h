#ifndef __KERNELS__H
#define __KERNELS__H
#include <stdio.h>

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
__device__ __forceinline__ void normalize_pixel_cuda(int32_t *target, int32_t pixel_idx, int32_t smallest,
					 int32_t largest) {
	if(smallest == largest) {
		return; // to prevent division by zero (see line below)
	}
  target[pixel_idx] = ((target[pixel_idx] - smallest) * 255) / (largest - smallest);
}

// applies the filter to the input image at the given row and column
// returns sum of filter application
__device__ __forceinline__ int apply_filter_cuda(const int32_t *input, const int8_t *filter, int32_t dimension, 
    int width, int height, int row, int col) {
    
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
                int32_t pixel = input[filter_idx];
                int8_t filter_value = filter[i * dimension + j];
                sum += pixel * filter_value;
            }
        }
    }
    return sum;
}

void run_kernel(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height, float *time_in, float *time_out, float *time_total);

__global__ void kernel5(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height);

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest);

#endif
