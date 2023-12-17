#include "kernel.h"
#include <cstdint>
#include "reduce.h"

template<int channels>
void run_kernel(const int8_t *filter, int32_t dimension, const Pixel<channels> *input,
                 Pixel<channels> *output, int32_t width, int32_t height,
                 int red, int green, int blue, int alpha) {
  // red, green, blue, alpha are scalars for how much of the filter we apply
  // onto the RGBA portions of the pixel

  int pixels = width * height;
  int blockSize;
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  int gridSize = (4 * height + blockSize - 1) / blockSize;

   // we will malloc one huge block instead of many small ones for performance
   // in batched_transfer
   // then we will set pointers to the correct locations in the block
  Pixel<channels> *batched_transfer;
  int transfer_size = 2 * pixels * sizeof(Pixel<channels>) + dimension * dimension * sizeof(int8_t);

  Pixel<channels> *device_input, *device_output;
  int8_t *device_filter;
  Pixel<channels> *h_smallest, *h_largest;

  h_smallest = (Pixel<channels>*) malloc(sizeof(Pixel<channels>));
  h_largest = (Pixel<channels>*) malloc(sizeof(Pixel<channels>));
  
  for(int i = 0; i < channels; ++i) {
    h_smallest->data[i] = 0;
    h_largest->data[i] = 255;
    // initialize smallest/largest variables to min/max for unsigned char size 
  }

  Pixel<channels> *d_smallest, *d_largest;

  // create copy of input, output on pinned memory on host
  Pixel<channels> *h_pinned_input, *h_pinned_output;
  cudaMallocHost(&h_pinned_input, pixels * sizeof(Pixel<channels>));
  cudaMallocHost(&h_pinned_output, pixels * sizeof(Pixel<channels>));
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);

  // malloc one huge block instead of many small ones for performance
  cudaMalloc(&batched_transfer, transfer_size);
  device_input = batched_transfer;
  device_output = batched_transfer + pixels;
  device_filter = (int8_t*) (batched_transfer + 2 * pixels);

  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));

  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda memcpys");

  kernel<<<gridSize, blockSize>>>(device_filter, dimension, device_input, device_output, width, height);

  CUDA_CHECK_ERROR("launching kernel3");

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda device synchronize after kernel3");

  reduce_max<<<gridSize, blockSize>>>(device_output, d_largest, pixels);
  reduce_min<<<gridSize, blockSize>>>(device_output, d_smallest, pixels);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("reduction");

  normalize<<<gridSize, blockSize>>>(device_output, width, height, d_smallest, d_largest);

  CUDA_CHECK_ERROR("normalize");

  cudaDeviceSynchronize();

  cudaMemcpy(h_pinned_output, device_output, pixels * sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("sync after normalize");

  cudaMemcpy(output, h_pinned_output, pixels * sizeof(int32_t), cudaMemcpyHostToHost);
  cudaFreeHost(h_pinned_input);
  cudaFreeHost(h_pinned_output);

  free(h_smallest); free(h_largest); 

  cudaFree(batched_transfer);
  cudaFree(d_smallest); cudaFree(d_largest);
}

__global__  void kernel(const int8_t *filter, int32_t dimension,
                        const int32_t *input, int32_t *output, int32_t width,
                        int32_t height) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  int row = tid / width;
  int col = tid % width;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int sum = apply_filter_cuda(input, filter, dimension, width, height, row, col);
    output[pixel_idx] = sum;
  }
}

__global__ void normalize(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    normalize_pixel_cuda(image, pixel_idx, *smallest, *biggest);
  }
}
