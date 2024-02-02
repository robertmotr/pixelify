#include "kernel.h"
#include "reduce.h"
#include "pixel.h"
#include "filter.h"

// explicit instantiations
#include "instantiations.cuh"

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                 Pixel<channels> *output, int width, int height, struct kernel_args extra) {

  filter *h_filter =            nullptr;
  filter*                       device_filter;
  int*                          device_filter_data;
  char*                         device_filter_name;
  int                           pixels = width * height;
  Pixel<channels>               *device_input, *device_output;
  Pixel<channels>               *d_largest, *d_smallest;
  Pixel<channels>               *h_pinned_input, *h_pinned_output;
  Pixel<channels>               *h_smallest, *h_largest;          
  int blockSize;
  int gridSize;

  if(strcmp(filter_name, "NULL") != 0) {         
    h_filter = create_filter_from_strength(filter_name, width, height, extra.filter_strength);
    if(h_filter == nullptr) {
      printf("Error: filter is null\n");
      exit(1);
    }
  } 


  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  assert(blockSize != 0);
  gridSize = (16 * height + blockSize - 1) / blockSize; 

  h_smallest = new Pixel<channels>(INT_MAX);
  h_largest = new Pixel<channels>(INT_MIN);
  // create copy of input, output on pinned memory on host
  cudaHostAlloc(&h_pinned_input, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped);
  cudaHostAlloc(&h_pinned_output, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped); // possible bug
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying input to pinned input");

  // MALLOCS ON DEVICE
  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda mallocs for input, output, largest, smallest");

  // HANDLE MALLOC AND MEMCPY FOR FILTER ONLY
  if(h_filter != nullptr && strcmp(filter_name, "NULL") != 0) {
    cudaMalloc(&device_filter, sizeof(filter));
    cudaMemcpy(&(device_filter->filter_dimension), &(h_filter->filter_dimension), sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->name_size), &(h_filter->name_size), sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&device_filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(int));
    cudaMemcpy(device_filter_data, h_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->filter_data), &device_filter_data, sizeof(int*), cudaMemcpyHostToDevice);

    cudaMalloc(&device_filter_name, h_filter->name_size * sizeof(char));
    cudaMemcpy(device_filter_name, h_filter->filter_name, h_filter->name_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->filter_name), &device_filter_name, sizeof(char*), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("cuda mallocs and memcpies for filter");
  }

  // MEMCPYS FROM HOST TO DEVICE
  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda memcpys and mallocs");

  // apply filter first if filter is not NULL
  // then apply everything else in the kernel_args struct
  // but first apply it filter_passes times
  for(int pass = 0; pass < extra.passes; pass++) {
    // run kernel pass times
  }
  // then apply everything else in the kernel_args struct

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
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying back d_smallest and d_largest");
  
  for(int ch = 0; ch < channels; ch++) {
    if(h_smallest->data[ch] < 0 || h_smallest->data[ch] > 255 ||
       h_largest->data[ch] < 0 || h_largest->data[ch] > 255) {
        normalize<channels><<<gridSize, blockSize>>>(device_output, width, height, d_smallest, d_largest, extra.normalize);
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
  delete h_smallest;
  delete h_largest;
  delete h_filter;
  cudaFree(device_filter);
  cudaFree(d_smallest); cudaFree(d_largest);
  cudaFree(device_input); cudaFree(device_output);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("freeing memory");
}

template<unsigned int channels>
__global__ void filter_kernel(const Pixel<channels> *in, Pixel<channels> *out, int width, int height,
                              const filter *filter, const struct kernel_args args) {

  extern __shared__ Pixel<channels> filter_smem[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  // load data and apply filter at same time
  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {

    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      filter_smem[pixel_idx].data[channel] = in[pixel_idx].data[channel];
      out[pixel_idx].data[channel] = apply_filter<channels>(filter_smem, filter, channel, width, height, row, col);
    }
  }
  __syncthreads();
}

template<unsigned int channels>
__global__  void other_kernel(const Pixel<channels> *in, Pixel<channels> *out, int width, int height,
                              unsigned char operation, struct kernel_args extra) {

  extern __shared__ Pixel<channels> smem[];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  // load data and apply operation at same time
  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      smem[pixel_idx].data[channel] = in[pixel_idx].data[channel];

      if(operation == OP_SHIFT_COLOURS) {
        out[pixel_idx].data[channel] = shift_colours(smem[pixel_idx].data[channel], extra, channel);
      } else if(operation == OP_BRIGHTNESS) {
        out[pixel_idx].data[channel] = smem[pixel_idx].data[channel] * (100 + extra.brightness) / 100;
      }
      else if(operation == OP_TINT) {
        out[pixel_idx].data[channel] = (1 - (float)(extra.blend_factor / 100)) * extra.tint[channel] + 
                                          (float)(extra.blend_factor / 100) * smem[pixel_idx].data[channel];
      }
    }
  }
}

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *largest, bool normalize_or_clamp) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;
  
  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    if(normalize_or_clamp) {
      normalize_pixel<channels>(target, pixel_idx, smallest, largest);
    } else {
      clamp_pixels<channels>(target, pixel_idx);
    }
  }
}