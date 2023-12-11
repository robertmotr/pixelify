#include "kernel.h"
#include <cstdint>
#include "reduce.h"

void run_kernel(const int8_t *filter, int32_t dimension, const int32_t *input,
                 int32_t *output, int32_t width, int32_t height) {
  // kernel5 uses a few optimizations that improves performance
  // overall computation approach is identical to kernel4
  // but we use pinned memory, malloc/transfer in and out in one batch
  // in addition we increase the grid size for better performance
  // and make a few optimizations to the kernel itself

  int pixels = width * height;
  int blockSize;
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  int gridSize = (4 * height + blockSize - 1) / blockSize;

   // we will malloc one huge block instead of many small ones for performance
   // in batched_transfer
   // then we will set pointers to the correct locations in the block
  int32_t *batched_transfer;
  int transfer_size = 2 * pixels * sizeof(int32_t) + dimension * dimension * sizeof(int8_t);

  int32_t *device_input, *device_output;
  int8_t *device_filter;
  int32_t *h_smallest, *h_largest;

  h_smallest = (int32_t*) malloc(sizeof(int32_t));
  h_largest = (int32_t*) malloc(sizeof(int32_t));
  *h_smallest = INT32_MAX;
  *h_largest = INT32_MIN;

  int32_t *d_smallest, *d_largest;

  // create copy of input, output on pinned memory on host
  int32_t *h_pinned_input, *h_pinned_output;
  cudaMallocHost(&h_pinned_input, pixels * sizeof(int32_t));
  cudaMallocHost(&h_pinned_output, pixels * sizeof(int32_t));
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(int32_t), cudaMemcpyHostToHost);

  cudaEvent_t start_in, stop_in, start_out, stop_out;
  cudaEventCreate(&start_in);
  cudaEventCreate(&stop_in);

  // measure transfer in
  cudaEventRecord(start_in);

  // malloc one huge block instead of many small ones for performance
  cudaMalloc(&batched_transfer, transfer_size);
  device_input = batched_transfer;
  device_output = batched_transfer + pixels;
  device_filter = (int8_t*) (batched_transfer + 2 * pixels);

  cudaMalloc(&d_smallest, sizeof(int32_t));
  cudaMalloc(&d_largest, sizeof(int32_t));

  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(int32_t), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda memcpys");

  kernel5<<<gridSize, blockSize>>>(device_filter, dimension, device_input, device_output, width, height);

  CUDA_CHECK_ERROR("launching kernel3");

  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR("cuda device synchronize after kernel3");

  reduce10_max<<<gridSize, blockSize>>>(device_output, d_largest, pixels);
  reduce10_min<<<gridSize, blockSize>>>(device_output, d_smallest, pixels);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("reduce10");

  normalize5<<<gridSize, blockSize>>>(device_output, width, height, d_smallest, d_largest);

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

__global__  void kernel5(const int8_t *filter, int32_t dimension,
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

__global__ void normalize5(int32_t *image, int32_t width, int32_t height,
                           int32_t *smallest, int32_t *biggest) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    normalize_pixel_cuda(image, pixel_idx, *smallest, *biggest);
  }
}
