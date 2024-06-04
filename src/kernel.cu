#include "kernel.cuh"
#include "pixel.h"
#include "filter_impl.h"
#include "filters.h"
#include <cuda_runtime.h>

__constant__ float                global_const_filter[MAX_FILTER_1D_SIZE];
__constant__ unsigned char        global_const_filter_dim;

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                Pixel<channels> *output, int width, int height,
                struct filter_args extra) {

  const filter *h_filter =                               nullptr;
  int                                                    pixels = width * height;
  Pixel<channels>                                        *device_output, *device_input;
  Pixel<channels>                                        *d_largest, *d_smallest;
  Pixel<channels>                                        *h_pinned_input, *h_pinned_output;
  Pixel<channels>                                        *h_smallest, *h_largest;          
  int blockSize  =                                       1024;
  int gridSize;
  h_smallest =                                           new Pixel<channels>{SHORT_MAX};
  h_largest =                                            new Pixel<channels>{SHORT_MIN};

  if(strcmp(filter_name, "NULL") != 0) {         
    h_filter = create_filter(filter_name, extra.dimension, extra.filter_strength);
    if(h_filter == nullptr) {
      printf("Error: filter is null\n");
      exit(1);
    }
  } 

  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  gridSize = (16 * height + blockSize - 1) / blockSize; 

  #ifdef _DEBUG
    printf("block size: %d\n", blockSize);
    assert(blockSize > 0);
    assert(gridSize > 0);
    printf("grid size: %d\n", gridSize);
  #endif

  // create copy of input, output on pinned memory on host
  cudaHostAlloc(&h_pinned_input, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped);
  cudaHostAlloc(&h_pinned_output, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped); 
  cudaMemcpyAsync(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  #ifdef _DEBUG
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("pinned memory");  
  #endif

  // MALLOCS ON DEVICE
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  #ifdef _DEBUG
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("mallocs on device");
    printf("size of image in bytes on h_pinned_input: %lu\n", pixels * sizeof(Pixel<channels>));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxSize = prop.maxTexture1D;
    printf("max size of 1d texture: %d\n", maxSize);
  #endif

  // copying filter data to constant memory
  cudaMemcpyToSymbol(global_const_filter, h_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(global_const_filter_dim, &h_filter->filter_dimension, sizeof(unsigned char), 0, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying to constant memory");

  // MEMCPYS FROM HOST TO DEVICE
  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying to device");

  // apply filter first if filter is not NULL
  // then apply everything else in the filter_args struct
  // but first apply it filter_passes times
  for(int pass = 0; pass < extra.passes; pass++) {
    #ifdef _DEBUG
      printf("applying filter_kernel on pass %d\n", pass);
    #endif
    filter_kernel<channels><<<gridSize, blockSize>>>(device_input, device_output, width, height, extra);
    CUDA_CHECK_ERROR("filter kernel pass");
  }
  // then apply everything else in the filter_args struct
  if(extra.alpha_shift != 0 || extra.red_shift != 0 || extra.green_shift != 0 || extra.blue_shift != 0) {
    shift_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    CUDA_CHECK_ERROR("shift colours");
  }
  if(extra.tint[0] != 0 || extra.tint[1] != 0 || extra.tint[2] != 0 || extra.tint[3] != 0) {
    tint_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    CUDA_CHECK_ERROR("tint");
  }
  if(extra.brightness != 0) {
    brightness_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    CUDA_CHECK_ERROR("brightness");
  }
  if(extra.invert) {
    invert_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    CUDA_CHECK_ERROR("invert");
  }

  // parallel reduction to find largest and smallest pixel values
  // for each channel respectively
  image_reduction<channels>(device_output, d_largest, pixels, MAX_REDUCE);
  image_reduction<channels>(device_output, d_smallest, pixels, MIN_REDUCE);
  // block size should be 1024 for optimal performance
  CUDA_CHECK_ERROR("reduction");

  // if d_largest or d_smallest are out of bounds
  // i.e outside of [0, 255] for any channel
  // then we need to normalize the image to bring it into valid bounds
  cudaMemcpy(h_smallest, d_smallest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_largest, d_largest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR("copying back d_smallest and d_largest");

  for(int ch = 0; ch < channels; ch++) {
    if(h_smallest->at(ch) < 0 || h_smallest->at(ch) > 255 ||
      h_largest->at(ch) < 0 || h_largest->at(ch) > 255) {
          normalize_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, d_smallest, d_largest, extra.normalize);
          CUDA_CHECK_ERROR("normalize");
          break;
    }
  }

  cudaMemcpy(output, device_output, pixels * sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpy(output, h_pinned_output, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  CUDA_CHECK_ERROR("copying back d_output to pinned output");

  // CLEANUP

  cudaFreeHost(h_pinned_input);
  cudaFreeHost(h_pinned_output);
  delete h_smallest;
  delete h_largest;
  cudaFree(d_smallest);
  cudaFree(d_largest);
  cudaFree(device_output); 
  // TODO: determine if this first run_kernel call and if so dont free device_output/device_input
  // we can reuse device_output/device_input for the next call because when we process the image
  // we copy back so we can just leave it there
  
  if(!(h_filter->properties->basic_filter)) {
    delete h_filter; // only delete if its NOT a basic filter
    // basic filters get reused
  }

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("freeing memory");
}

// applies the filter to the input image at the given row and column
// returns sum of filter application
template<unsigned int channels>
__device__ __forceinline__ short apply_filter(const Pixel<channels> __restrict_arr *device_input, unsigned int mask, int width, 
                                            int height, int row, int col) {
    #ifdef _DEBUG
      assert(device_input != nullptr);
      assert(mask < channels);
      assert(find_index(width, height, row, col) >= 0);
      assert(find_index(width, height, row, col) < width * height);
    #endif
      
    const __restrict_arr volatile float *const_filter = global_const_filter;
    const __restrict_arr volatile unsigned char const_filter_dim = global_const_filter_dim;

    float sum = 0;
    int start_i = row - const_filter_dim;
    int start_j = col - const_filter_dim;

    // iterate over the filter
    #pragma unroll
    for (int i = 0; i < const_filter_dim; i++) {
        #pragma unroll
        for (int j = 0; j < const_filter_dim; j++) {

            int filter_x = start_i + i;
            int filter_y = start_j + j;

            int idx = find_index(width, height, filter_x, filter_y);
            short member_value = __ldcs(device_input[idx].at_ptr(mask)); // fast load using constant memory

            float filter_value = const_filter[i * const_filter_dim + j];
            sum = __fmaf_rn(member_value, filter_value, sum); // fast multiply and add
        }
    }
    return (short) sum;
}

// main kernel that applies the filter to the input image
template<unsigned int channels>
__global__ void filter_kernel(const __restrict_arr Pixel<channels> *in, Pixel<channels> *out, int width, int height,
                              const struct filter_args args) {
  #ifdef _DEBUG
    assert(out != nullptr);
  #endif                      
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #pragma unroll
  for (int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx * __fdividef(1.0f, width); // fast divide by width
    int col = pixel_idx % width;

    #pragma unroll
    for(int ch = 0; ch < channels; ch++) {
      out[pixel_idx].set(ch, apply_filter<channels>(in, ch, width, height,
                                                      row, col)); 
    }  
  } 
}

// shifts the colours of the input image as cuda kernel
template<unsigned int channels>
__global__ void shift_kernel(Pixel<channels> *d_pixels, int width, int height,
                             const struct filter_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(d_pixels != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = d_pixels[pixel_idx].at(channel);
      d_pixels[pixel_idx].set(channel, shift_colours(channel_val, extra, channel));
    }
  }
}

// applies the brightness filter to the input image as cuda kernel
template<unsigned int channels>
__global__ void brightness_kernel(Pixel<channels> *d_pixels, int width, int height,
                                  const struct filter_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(d_pixels != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = d_pixels[pixel_idx].at(channel);
      d_pixels[pixel_idx].set(channel, channel_val * (100 + extra.brightness) / 100);
    }
  }
}

// applies the tint filter to the input image as cuda kernel
template<unsigned int channels>
__global__ void tint_kernel(Pixel<channels> *d_pixels, int width, int height,
                            const struct filter_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(d_pixels != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = d_pixels[pixel_idx].at(channel);
      d_pixels[pixel_idx].set(channel, (1 - (float)(extra.blend_factor)) * extra.tint[channel] + 
                                      (float)(extra.blend_factor) * channel_val);
    }
  }
}

// inverts the input image as cuda kernel
template<unsigned int channels>
__global__ void invert_kernel(Pixel<channels> *d_pixels, int width, int height,
                              const struct filter_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(d_pixels != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = d_pixels[pixel_idx].at(channel);
      d_pixels[pixel_idx].set(channel, 255 - channel_val);
    }
  }
}

// normalizes the input image to the range [0, 255] as cuda kernel
template<unsigned int channels>
__global__ void normalize_kernel(Pixel<channels> *target, int width, int height,
                          const Pixel<channels> __restrict_arr *smallest,
                          const Pixel<channels> __restrict_arr *largest,
                          bool normalize) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  #ifdef _DEBUG
    assert(target != nullptr);
    assert(smallest != nullptr);
    assert(largest != nullptr);
    printf("smallest: %d %d %d %d\n", smallest->at(0), smallest->at(1), smallest->at(2), smallest->at(3));
    printf("largest: %d %d %d %d\n", largest->at(0), largest->at(1), largest->at(2), largest->at(3));
    printf("Normalize: %d\n", normalize);
  #endif
  
  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += blockDim.x * gridDim.x) {
    if(normalize) {
      normalize_pixel<channels>(target, pixel_idx, smallest, largest);
    } else {
      clamp_pixels<channels>(target, pixel_idx);
    }
  }
}

// Warp reduction helper for finding the largest and smallest pixel values
template <unsigned int blockSize, unsigned int channels>
__device__ void warp_reduce_pixels(volatile Pixel<channels> *sdata, unsigned int tid, bool reduce_type) {
    // we use conditional statements to avoid ancillary instructions and thus improve performance
    if (blockSize >= 64) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 32].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 32].at(ch)));
        }
    }
    if (blockSize >= 32) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 16].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 16].at(ch)));
        }
    }
    if (blockSize >= 16) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 8].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 8].at(ch)));
        }
    }
    if (blockSize >= 8) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 4].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 4].at(ch)));
        }
    }
    if (blockSize >= 4) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 2].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 2].at(ch)));
        }
    }
    if (blockSize >= 2) {
        for (int ch = 0; ch < channels; ch++) {
            sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + 1].at(ch)) : 
                                                            min(sdata[tid].at(ch), sdata[tid + 1].at(ch)));
        }
    }
}


// Image reduction kernel for finding the largest and smallest pixel values
template<unsigned int channels, unsigned int blockSize>
__global__ void image_reduction_kernel(const Pixel<channels> *d_image, Pixel<channels> *d_out, int pixels, bool reduce_type) {
  extern __shared__ Pixel<channels> sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  // initialize shared memory
  sdata[tid] = (reduce_type == MAX_REDUCE) ? Pixel<channels>{SHORT_MIN} : Pixel<channels>{SHORT_MAX};


  // load two pixels per thread from global memory
  #pragma unroll
  while (i < pixels) {
    #pragma unroll
    for(int ch = 0; ch < channels; ch++) {
      sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), d_image[i].at(ch)) : 
                                                      min(sdata[tid].at(ch), d_image[i].at(ch)));
    }
    if (i + blockSize < pixels) {
      #pragma unroll
      for(int ch = 0; ch < channels; ch++) {
        sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), d_image[i + blockSize].at(ch)) : 
                                                        min(sdata[tid].at(ch), d_image[i + blockSize].at(ch)));
      }
    }
    i += gridSize; 
  }
  __syncthreads(); // ensure all threads have loaded their data

  // conditional reductions for larger blocks
  // continue reducing pixels in shared memory and sync after each step
  #pragma unroll
  for(int s = blockSize / 2; s > 32; s >>= 1) {
    if (tid < s) {
      #pragma unroll
      for(int ch = 0; ch < channels; ch++) {
        sdata[tid].set(ch, (reduce_type == MAX_REDUCE) ? max(sdata[tid].at(ch), sdata[tid + s].at(ch)) : 
                                                        min(sdata[tid].at(ch), sdata[tid + s].at(ch)));
      }
    }
    __syncthreads();
  }

  if (tid < 32) {
    warp_reduce_pixels<blockSize, channels>(sdata, tid, reduce_type);
  }
  
  if (tid == 0) {
    #pragma unroll
    for(int ch = 0; ch < channels; ch++) {
      d_out[blockIdx.x]->set(ch, sdata[0].at(ch));
    }
  }
}

// Host function for performing image reduction
template <unsigned int channels>
void image_reduction(const Pixel<channels> *d_image, Pixel<channels> *d_result, int pixels, bool reduce_type) {
  int blockSize = 1024;
  int gridSize = (pixels + blockSize * 2 - 1) / (blockSize * 2);

  // debugging output
  #ifdef _DEBUG
    printf("block size: %d\n", blockSize);
    printf("grid size: %d\n", gridSize);
    assert(blockSize > 0);
    assert(gridSize > 0);
  #endif

  image_reduction_kernel<channels, 1024><<<gridSize, blockSize, blockSize * sizeof(Pixel<channels>)>>>(d_image, d_result, pixels, reduce_type);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("first non-recursive reduction");

  // call the kernel recursively 
  if (gridSize > 1) {
    Pixel<channels> *d_intermediate;
    cudaMalloc(&d_intermediate, gridSize * sizeof(Pixel<channels>));
    image_reduction<channels>(d_result, d_intermediate, gridSize, reduce_type);
    cudaFree(d_intermediate);
  }
}

// EXPLICIT INSTANTIATIONS: 
template void run_kernel(const char *filter_name, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height, struct filter_args extra);

template void run_kernel(const char *filter_name, const Pixel<4u> *input, 
                 Pixel<4u> *output, int width, int height, struct filter_args extra);

template __global__ void shift_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                         const struct filter_args extra);

template __global__ void shift_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                          const struct filter_args extra);

template __device__ __forceinline__ short apply_filter<3u>(const Pixel<3u> *device_input,
                                                           unsigned int mask, int width, int height, int row, int col);

template __device__ __forceinline__ short apply_filter<4u>(const Pixel<4u> *device_input, 
                                                           unsigned int mask, int width, int height, int row, int col);

template __global__ void normalize_kernel<3u>(Pixel<3u> *target, int width, int height,
                          const Pixel<3u> __restrict_arr *smallest,
                          const Pixel<3u> __restrict_arr *largest,
                          bool normalize);

template __global__ void normalize_kernel<4u>(Pixel<4u> *target, int width, int height,
                          const Pixel<4u> __restrict_arr *smallest,
                          const Pixel<4u> __restrict_arr *largest,
                          bool normalize);        

template __global__ void filter_kernel<3u>(const Pixel<3u> *d_in, Pixel<3u> *out, int width, int height,
                                           const struct filter_args args);

template __global__ void filter_kernel<4u>(const Pixel<4u> *d_in, Pixel<4u> *out, int width, int height,
                                           const struct filter_args args);       

template __global__ void brightness_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                               const struct filter_args extra);

template __global__ void brightness_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                                const struct filter_args extra);        

template __global__ void tint_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                        const struct filter_args extra);  
                                      
template __global__ void tint_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                         const struct filter_args extra);

template __global__ void invert_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                           const struct filter_args extra);

template __global__ void invert_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                           const struct filter_args extra);      

template void image_reduction<3u>(const Pixel<3u> *d_image, Pixel<3u> *d_result, int pixels, bool reduce_type);

template void image_reduction<4u>(const Pixel<4u> *d_image, Pixel<4u> *d_result, int pixels, bool reduce_type);  