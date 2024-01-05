#include "kernel.h"
#include <cstdint>
#include "reduce.h"
#include "pixel.h"

template<unsigned int channels>
void run_kernel(const int8_t *filter, int32_t dimension, const Pixel<channels> *input,
                 Pixel<channels> *output, int32_t width, int32_t height) {
  int pixels = width * height;
  int blockSize;
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  int gridSize = (4 * height + blockSize - 1) / blockSize;

  Pixel<channels> *h_pinned_input, *h_pinned_output;
  // create copy of input, output on pinned memory on host
  cudaMallocHost(&h_pinned_input, pixels * sizeof(Pixel<channels>));
  cudaMallocHost(&h_pinned_output, pixels * sizeof(Pixel<channels>));
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);

  Pixel<channels> *device_input, *device_output;
  Pixel<channels> *d_largest, *d_smallest;
  int8_t *device_filter;

  Pixel<channels> *h_smallest, *h_largest;

  h_smallest = (Pixel<channels>*) malloc(sizeof(Pixel<channels>));
  h_largest = (Pixel<channels>*) malloc(sizeof(Pixel<channels>));
  
  for(int ch = 0; ch < channels; ch++) {
    h_smallest->data[ch] = PIXEL_NULL_CHANNEL;
    h_largest->data[ch] = PIXEL_NULL_CHANNEL;
  }

  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  
  cudaMalloc(&device_filter, dimension * dimension * sizeof(int8_t));
  cudaMemcpy(device_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);

  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));

  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda memcpys");

  kernel<channels><<<gridSize, BLOCK_SIZE>>>(device_filter, dimension, device_input, device_output, width, height);

  CUDA_CHECK_ERROR("launching kernel3");

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda device synchronize after kernel3");

  // parallel reduction to find largest and smallest pixel values
  // for each channel respectively
  image_reduction<channels>(device_output, d_largest, pixels, MAX_REDUCE);
  image_reduction<channels>(device_output, d_smallest, pixels, MIN_REDUCE);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("reduction");

  // now normalize the image
  normalize<channels><<<gridSize, BLOCK_SIZE>>>(device_output, width, height, d_smallest, d_largest);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("normalize");

  cudaMemcpy(h_pinned_output, device_output, pixels * sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("sync after normalize");

  cudaMemcpy(output, h_pinned_output, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  cudaFreeHost(h_pinned_input);
  cudaFreeHost(h_pinned_output);

  free(h_smallest); free(h_largest); 

  cudaFree(device_filter);
  cudaFree(d_smallest); cudaFree(d_largest);
}

template<unsigned int channels>
__global__  void kernel(const int8_t *filter, int32_t dimension,
                        const Pixel<channels> *input, Pixel<channels> *output, int32_t width, int32_t height) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  int row = tid / width;
  int col = tid % width;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    for(int channel = 0; channel < 3; channel++) {
      int sum = apply_filter_cuda<channels>(input, filter, channel, dimension, width, height, row, col);
      output[pixel_idx].data[channel] = sum;
    }
  }
}

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int32_t width, int32_t height,
                           Pixel<channels> *smallest, Pixel<channels> *largest) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    normalize_pixel_cuda<channels>(target, pixel_idx, smallest, largest);
  }
}

// explicitly instantiate
template void run_kernel<3u>(const int8_t *filter, int32_t dimension, const Pixel<3u> *input,
                 Pixel<3u> *output, int32_t width, int32_t height);

template void run_kernel<4u>(const int8_t *filter, int32_t dimension, const Pixel<4u> *input,
                  Pixel<4u> *output, int32_t width, int32_t height);

template __global__ void kernel<3u>(const int8_t *filter, int32_t dimension,
                        const Pixel<3u> *input, Pixel<3u> *output, int32_t width, int32_t height);

template __global__ void kernel<4u>(const int8_t *filter, int32_t dimension,
                        const Pixel<4u> *input, Pixel<4u> *output, int32_t width, int32_t height);

template __global__ void normalize<3u>(Pixel<3u> *target, int32_t width, int32_t height,
                           Pixel<3u> *smallest, Pixel<3u> *largest);

template __global__ void normalize<4u>(Pixel<4u> *target, int32_t width, int32_t height,
                            Pixel<4u> *smallest, Pixel<4u> *largest);

template void image_reduction<3u>(Pixel<3u> *d_input, Pixel<3u>* d_result, unsigned int size, bool op);

template void image_reduction<4u>(Pixel<4u> *d_input, Pixel<4u>* d_result, unsigned int size, bool op);