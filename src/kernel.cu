#include "kernel.cuh"
#include "reduce.cuh"
#include "pixel.h"
#include "filter_impl.h"
#include "filters.h"
#include <cuda_runtime.h>

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                Pixel<channels> *output, int width, int height,
                struct kernel_args extra) {

  const size_t src_pitch =                               width * sizeof(Pixel<channels>);
  const filter *h_filter =                               nullptr;
  filter*                                                device_filter;
  int*                                                   device_filter_data;
  char*                                                  device_filter_name;
  int                                                    pixels = width * height;
  Pixel<channels>                                        *device_output;
  Pixel<channels>                                        *d_largest, *d_smallest;
  Pixel<channels>                                        *h_pinned_input, *h_pinned_output;
  Pixel<channels>                                        *h_smallest, *h_largest;          
  int blockSize;
  int gridSize;
  cudaArray_t cu_array;
  cudaTextureObject_t tex_obj =                          0;
  h_smallest =                                           new Pixel<channels>(SHORT_MAX);
  h_largest =                                            new Pixel<channels>(SHORT_MIN);

  if(strcmp(filter_name, "NULL") != 0) {         
    h_filter = create_filter(filter_name, extra.dimension, extra.filter_strength);
    if(h_filter == nullptr) {
      printf("Error: filter is null\n");
      exit(1);
    }
  } 

  cudaDeviceGetAttribute(&blockSize, cudaDevAttrMaxThreadsPerBlock, 0);
  #ifdef _DEBUG
    printf("block size: %d\n", blockSize);
    assert(blockSize > 0);
  #endif
  gridSize = (8 * height + blockSize - 1) / blockSize; 

  // create copy of input, output on pinned memory on host
  cudaHostAlloc(&h_pinned_input, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped);
  cudaHostAlloc(&h_pinned_output, pixels * sizeof(Pixel<channels>), cudaHostAllocMapped); // possible bug
  cudaMemcpy(h_pinned_input, input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying input to pinned input");

  // MALLOCS ON DEVICE
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda mallocs for input, output, largest, smallest");
  // END MALLOCS

  // -- SETTING UP STUFF FOR TEXTURE -- 
  cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8 * sizeof(short), 
                                                             8 * sizeof(short), 
                                                             8 * sizeof(short),
                                                             8 * sizeof(short), 
                                                             cudaChannelFormatKindSigned); 

  cudaMallocArray(&cu_array, &channel_desc, width, height);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda malloc array");

  cudaMemcpy2DToArray(cu_array, 0, 0, h_pinned_input, src_pitch, width * sizeof(Pixel<channels>), (size_t) height, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("2d array copy to device_input");

  struct cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = cu_array;
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("resource desc");

  struct cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = cudaAddressModeBorder;
  tex_desc.addressMode[1] = cudaAddressModeBorder;
  tex_desc.filterMode = cudaFilterModePoint;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 0;
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("texture desc");

  // texture object
  cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("creating texture object");
  // -- END TEXTURE SETUP --

  // HANDLE MALLOC AND MEMCPY FOR FILTER ONLY
  if(h_filter != nullptr && strcmp(filter_name, "NULL") != 0) {
    cudaMalloc(&device_filter, sizeof(filter));
    cudaMemcpy(&(device_filter->filter_dimension), &(h_filter->filter_dimension), sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->name_size), &(h_filter->name_size), sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc(&device_filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(unsigned int));
    cudaMemcpy(device_filter_data, h_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->filter_data), &device_filter_data, sizeof(float*), cudaMemcpyHostToDevice);

    cudaMalloc(&device_filter_name, h_filter->name_size * sizeof(char));
    cudaMemcpy(device_filter_name, h_filter->filter_name, h_filter->name_size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_filter->filter_name), &device_filter_name, sizeof(char*), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("cuda mallocs and memcpies for filter");
  }
  // END FILTER SETUP

  // MEMCPYS FROM HOST TO DEVICE
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda memcpys and mallocs");

  // apply filter first if filter is not NULL
  // then apply everything else in the kernel_args struct
  // but first apply it filter_passes times
  for(int pass = 0; pass < extra.passes; pass++) {
    #ifdef _DEBUG
      printf("applying filter_kernel");
    #endif

    filter_kernel<channels><<<gridSize, blockSize, sizeof(float) * h_filter->filter_dimension * h_filter->filter_dimension>>>(tex_obj, device_output,
                                                                                      width, height, device_filter, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("filter kernel");
    cudaMemcpy2DToArray(cu_array, 0, 0, device_output, src_pitch, src_pitch, (size_t) height, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("2d array copy back to device_output");
  }
  // then apply everything else in the kernel_args struct
  if(extra.alpha_shift != 0 || extra.red_shift != 0 || extra.green_shift != 0 || extra.blue_shift != 0) {
    shift_kernel<channels><<<gridSize, blockSize>>>(tex_obj, device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("shift colours");
    cudaMemcpy2DToArray(cu_array, 0, 0, device_output, src_pitch, src_pitch, (size_t) height, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("2d array copy back to device_output");
  }
  if(extra.tint[0] != 0 || extra.tint[1] != 0 || extra.tint[2] != 0 || extra.tint[3] != 0) {
    tint_kernel<channels><<<gridSize, blockSize>>>(tex_obj, device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("tint");
    cudaMemcpy2DToArray(cu_array, 0, 0, device_output, src_pitch, src_pitch, (size_t) height, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("2d array copy back to device_output");
  }
  if(extra.brightness != 0) {
    brightness_kernel<channels><<<gridSize, blockSize>>>(tex_obj, device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("brightness");
    cudaMemcpy2DToArray(cu_array, 0, 0, device_output, src_pitch, src_pitch, (size_t) height, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("2d array copy back to device_output");
  }
  if(extra.invert) {
    invert_kernel<channels><<<gridSize, blockSize>>>(tex_obj, device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("invert");
    cudaMemcpy2DToArray(cu_array, 0, 0, device_output, src_pitch, src_pitch, (size_t) height, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("2d array copy back to device_output");
  }

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
  cudaDestroyTextureObject(tex_obj);
  cudaFreeArray(cu_array);

  cudaFreeHost(h_pinned_input); cudaFreeHost(h_pinned_output);
  delete h_smallest;
  delete h_largest;
  cudaFree(device_filter);
  cudaFree(d_smallest); cudaFree(d_largest);
  cudaFree(device_output);
  
  if(!(h_filter->properties->basic_filter)) {
    delete h_filter; // only delete if its NOT a basic filter
    // basic filters get reused
  }

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("freeing memory");
}

template<unsigned int channels>
__global__ void filter_kernel(const cudaTextureObject_t tex_obj, Pixel<channels> *out, int width, int height,
                              const filter *filter, const struct kernel_args args) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(tex_obj != 0);
    assert(out != nullptr);
    assert(filter != nullptr);
  #endif

  extern __constant__ float const_filter[];

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int ch = 0; ch < channels; ch++) {
      out[pixel_idx].data[ch] = apply_filter<channels>(tex_obj, filter, ch, width, height,
                                                       row, col);
    }  
  } 
}

template<unsigned int channels>
__global__ void shift_kernel(const cudaTextureObject_t in, Pixel<channels> *out, int width, int height,
                            struct kernel_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(in != 0);
    assert(out != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = get_texel<channels>(in, row, col, channel);
      out[pixel_idx].data[channel] = shift_colours(channel_val, extra, channel);
    }
  }
}

template<unsigned int channels>
__global__ void brightness_kernel(const cudaTextureObject_t in, Pixel<channels> *out, int width, int height,
                                  struct kernel_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(in != 0);
    assert(out != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = get_texel<channels>(in, row, col, channel);
      out[pixel_idx].data[channel] = channel_val * (100 + extra.brightness) / 100;
    }
  }
}

template<unsigned int channels>
__global__ void tint_kernel(const cudaTextureObject_t in, Pixel<channels> *out, int width, int height,
                            struct kernel_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(in != 0);
    assert(out != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = get_texel<channels>(in, row, col, channel);
      out[pixel_idx].data[channel] = (1 - (float)(extra.blend_factor)) * extra.tint[channel] + 
                                      (float)(extra.blend_factor) * channel_val;
    }
  }
}

template<unsigned int channels>
__global__ void invert_kernel(const cudaTextureObject_t in, Pixel<channels> *out, int width, int height,
                              struct kernel_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(in != 0);
    assert(out != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = get_texel<channels>(in, row, col, channel);
      out[pixel_idx].data[channel] = 255 - channel_val;
    }
  }
}

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *largest,
                           bool normalize) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  #ifdef _DEBUG
    assert(target != nullptr);
    assert(smallest != nullptr);
    assert(largest != nullptr);
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

// EXPLICIT INSTANTIATIONS:
template void run_kernel(const char *filter_name, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height, struct kernel_args extra);

template void run_kernel(const char *filter_name, const Pixel<4u> *input, 
                 Pixel<4u> *output, int width, int height, struct kernel_args extra);

template __global__ void shift_kernel<3u>(const cudaTextureObject_t in, Pixel<3u> *out, int width, int height,
                                         struct kernel_args extra);

template __global__ void shift_kernel<4u>(const cudaTextureObject_t in, Pixel<4u> *out, int width, int height,
                                          struct kernel_args extra);

template __global__ void filter_kernel<3u>(const cudaTextureObject_t tex_obj, Pixel<3u> *out, int width, int height,
                                           const filter *filter, const struct kernel_args args);

template __global__ void filter_kernel<4u>(const cudaTextureObject_t tex_obj, Pixel<4u> *out, int width, int height,
                                            const filter *filter, const struct kernel_args args);       

template __global__ void brightness_kernel<3u>(const cudaTextureObject_t in, Pixel<3u> *out, int width, int height,
                                               struct kernel_args extra);

template __global__ void brightness_kernel<4u>(const cudaTextureObject_t in, Pixel<4u> *out, int width, int height,
                                                struct kernel_args extra);        

template __global__ void tint_kernel<3u>(const cudaTextureObject_t in, Pixel<3u> *out, int width, int height,
                                        struct kernel_args extra);  
                                      
template __global__ void tint_kernel<4u>(const cudaTextureObject_t in, Pixel<4u> *out, int width, int height,
                                         struct kernel_args extra);

template __global__ void invert_kernel<3u>(const cudaTextureObject_t in, Pixel<3u> *out, int width, int height,
                                           struct kernel_args extra);

template __global__ void invert_kernel<4u>(const cudaTextureObject_t in, Pixel<4u> *out, int width, int height,
                                            struct kernel_args extra);      