#include "include/kernel.h"
#include "include/reduce.h"
#include "include/pixel.h"

template<unsigned int channels>
void run_kernel(const int8_t *filter, int dimension, const Pixel<channels> *input,
                 Pixel<channels> *output, int width, int height) {
  int pixels = width * height;
  int blockSize;
  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  int gridSize = (4 * height + blockSize - 1) / blockSize;

  Pixel<channels> *h_pinned_input, *h_pinned_output;
  Pixel<channels> *h_smallest, *h_largest;

  h_smallest = new Pixel<channels>(INT_MAX);
  h_largest = new Pixel<channels>(INT_MIN);
  // create copy of input, output on pinned memory on host
  cudaMallocHost(&h_pinned_input, pixels * sizeof(Pixel<channels>));
  cudaMallocHost(&h_pinned_output, pixels * sizeof(Pixel<channels>));
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);

  Pixel<channels> *device_input, *device_output;
  Pixel<channels> *d_largest, *d_smallest;
  int8_t *device_filter;

  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_filter, dimension * dimension * sizeof(int8_t));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));

  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(device_filter, filter, dimension * dimension * sizeof(int8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda memcpys and mallocs");

  kernel<channels><<<gridSize, blockSize>>>(device_filter, dimension, device_input, device_output, width, height);
  CUDA_CHECK_ERROR("launching kernel3");
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda device synchronize after kernel3");

  // parallel reduction to find largest and smallest pixel values
  // for each channel respectively
  image_reduction<channels>(device_output, d_largest, pixels, MAX_REDUCE);
  image_reduction<channels>(device_output, d_smallest, pixels, MIN_REDUCE);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("reduction");

  // if d_largest or d_smallest are out of bounds
  // i.e outside of [0, 255] for any channel
  // then we need to normalize the image to bring it into valid bounds
  cudaMemcpy(h_smallest, d_smallest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_largest, d_largest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  
  for(int ch = 0; ch < channels; ch++) {
    if(h_smallest->data[ch] < 0 || h_smallest->data[ch] > 255 ||
       h_largest->data[ch] < 0 || h_largest->data[ch] > 255) {
        printf("-----normalizing-----\n");
        normalize<channels><<<gridSize, blockSize>>>(device_output, width, height, d_smallest, d_largest);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("normalize");
        break;
    }
  }

  cudaMemcpy(h_pinned_output, device_output, pixels * sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpy(output, h_pinned_output, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying back d_output to pinned output");

  // cleanup
  cudaFreeHost(h_pinned_input); cudaFreeHost(h_pinned_output);
  free(h_smallest); free(h_largest); 
  cudaFree(device_filter);
  cudaFree(d_smallest); cudaFree(d_largest);
  cudaFree(device_input); cudaFree(device_output);
}

template<unsigned int channels>
__global__  void kernel(const int8_t *filter, int dimension,
                        const Pixel<channels> *input, Pixel<channels> *output, int width, int height) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      int sum = apply_filter<channels>(input, filter, channel, dimension, width, height, row, col);
      output[pixel_idx].data[channel] = sum;
    }
  }
}

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *largest) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    normalize_pixel<channels>(target, pixel_idx, smallest, largest);
  }
}

// explicitly instantiate
template void run_kernel<3u>(const int8_t *filter, int dimension, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height);

template void run_kernel<4u>(const int8_t *filter, int dimension, const Pixel<4u> *input,
                  Pixel<4u> *output, int width, int height);

template __global__ void kernel<3u>(const int8_t *filter, int dimension,
                        const Pixel<3u> *input, Pixel<3u> *output, int width, int height);

template __global__ void kernel<4u>(const int8_t *filter, int dimension,
                        const Pixel<4u> *input, Pixel<4u> *output, int width, int height);

template __global__ void normalize<3u>(Pixel<3u> *target, int width, int height,
                           const Pixel<3u> *smallest, const Pixel<3u> *largest);

template __global__ void normalize<4u>(Pixel<4u> *target, int width, int height,
                            const Pixel<4u> *smallest, const Pixel<4u> *largest);

template void image_reduction<3u>(const Pixel<3u> *d_input, Pixel<3u>* d_result, unsigned int size, bool op);

template void image_reduction<4u>(const Pixel<4u> *d_input, Pixel<4u>* d_result, unsigned int size, bool op);