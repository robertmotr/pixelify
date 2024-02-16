#include "kernel.cuh"
#include "reduce.cuh"
#include "pixel.h"
#include "filter_impl.h"
#include "filters.h"
#include <cuda_runtime.h>

__constant__ float                global_const_filter[MAX_FILTER_1D_SIZE];
__constant__ unsigned char        global_const_filter_dim;

// template<unsigned int channels>
// void create_texture_object(void *pinned_input, int width, int height, cudaArray_t* cu_array,
//                            cudaTextureObject_t* tex_obj) {
//   cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(8 * sizeof(short), 
//                                                               8 * sizeof(short), 
//                                                               8 * sizeof(short),
//                                                               8 * sizeof(short), 
//                                                               cudaChannelFormatKindSigned); 

  
//   Pixel<channels> *h_pinned_input = (Pixel<channels> *)pinned_input;
  
//   cudaMallocArray(cu_array, &channel_desc, width, height);
//   cudaDeviceSynchronize();
//   CUDA_CHECK_ERROR("cuda malloc array");

//   cudaMemcpyToArray(*cu_array, 0, 0, h_pinned_input, width * height * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
//   cudaDeviceSynchronize();
//   CUDA_CHECK_ERROR("2d array copy to device_input");

//   struct cudaResourceDesc res_desc;
//   memset(&res_desc, 0, sizeof(res_desc));
//   res_desc.resType = cudaResourceTypeArray;
//   res_desc.res.array.array = *cu_array;
//   cudaDeviceSynchronize();
//   CUDA_CHECK_ERROR("resource desc");

//   struct cudaTextureDesc tex_desc;
//   memset(&tex_desc, 0, sizeof(tex_desc));
//   tex_desc.addressMode[0] = cudaAddressModeBorder;
//   tex_desc.filterMode = cudaFilterModePoint;
//   tex_desc.readMode = cudaReadModeElementType;
//   tex_desc.normalizedCoords = 0;
//   cudaDeviceSynchronize();
//   CUDA_CHECK_ERROR("texture desc");

//   // texture object
//   cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);
//   CUDA_CHECK_ERROR("creating texture object");
// }

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
  int blockSize;
  int gridSize;
  // cudaArray_t cu_array;
  // cudaTextureObject_t tex_obj =                          0;
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

  // MALLOCS ON DEVICE
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));
  // END MALLOCS

  #ifdef _DEBUG
    printf("size of image in bytes on h_pinned_input: %lu\n", pixels * sizeof(Pixel<channels>));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxSize = prop.maxTexture1D;
    printf("max size of 1d texture: %d\n", maxSize);
  #endif

  // // -- SETTING UP STUFF FOR TEXTURE -- 
  // create_texture_object<channels>((void*)h_pinned_input, width, height, &cu_array, &tex_obj);

  // HANDLE MALLOC AND MEMCPY FOR FILTER ONLY
  cudaMemcpyToSymbolAsync(global_const_filter, h_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbolAsync(global_const_filter_dim, &h_filter->filter_dimension, sizeof(unsigned char), 0, cudaMemcpyHostToDevice);
  // END FILTER SETUP

  // MEMCPYS FROM HOST TO DEVICE
  cudaMemcpyAsync(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying to device");

  // apply filter first if filter is not NULL
  // then apply everything else in the filter_args struct
  // but first apply it filter_passes times
  for(int pass = 0; pass < extra.passes; pass++) {
    #ifdef _DEBUG
      printf("applying filter_kernel on pass %d\n", pass);
    #endif
    if(pass == 0) {
      filter_kernel<channels><<<gridSize, blockSize>>>(device_input, device_output, width, height, extra);
    }
    else {
      filter_kernel<channels><<<gridSize, blockSize>>>(device_input, device_output, width, height, extra);
    }
    cudaMemcpyAsync(device_input, device_output, pixels * sizeof(Pixel<channels>), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("filter kernel pass");
  }
  // then apply everything else in the filter_args struct
  if(extra.alpha_shift != 0 || extra.red_shift != 0 || extra.green_shift != 0 || extra.blue_shift != 0) {
    shift_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("shift colours");
  }
  if(extra.tint[0] != 0 || extra.tint[1] != 0 || extra.tint[2] != 0 || extra.tint[3] != 0) {
    tint_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("tint");
    cudaDeviceSynchronize();
  }
  if(extra.brightness != 0) {
    brightness_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("brightness");
    cudaDeviceSynchronize();
  }
  if(extra.invert) {
    invert_kernel<channels><<<gridSize, blockSize>>>(device_output, width, height, extra);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("invert");
    cudaDeviceSynchronize();
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
  cudaMemcpyAsync(h_smallest, d_smallest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(h_largest, d_largest, sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
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

  cudaMemcpyAsync(h_pinned_output, device_output, pixels * sizeof(Pixel<channels>), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(output, h_pinned_output, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToHost);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("copying back d_output to pinned output");
  
  // cleanup
  // cudaDestroyTextureObject(tex_obj);
  // cudaFreeArray(cu_array);

  cudaFreeHost(h_pinned_input); cudaFreeHost(h_pinned_output);
  delete h_smallest;
  delete h_largest;
  cudaFree(d_smallest); cudaFree(d_largest);
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
__device__ __forceinline__ short apply_filter(const Pixel<channels> *device_input, unsigned int mask, int width, 
                                            int height, int row, int col) {
    #ifdef _DEBUG
      assert(device_input != nullptr);
      assert(mask < channels);
      assert(find_index(width, height, row, col) >= 0);
      assert(find_index(width, height, row, col) < width * height);
    #endif
      
    const __restrict_arr float *const_filter = global_const_filter;
    const __restrict_arr unsigned char const_filter_dim = global_const_filter_dim;

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
            short member_value = __ldg(device_input[idx]->at(mask));

            float filter_value = const_filter[i * const_filter_dim + j];
            sum = __fmaf_rn(member_value, filter_value, sum);
        }
    }
    return (short) sum;
}

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
    int row = pixel_idx * __fdividef(1.0f, width);
    int col = pixel_idx % width;

    #pragma unroll
    for(int ch = 0; ch < channels; ch++) {
      out[pixel_idx]->set(ch, apply_filter<channels>(in, ch, width, height,
                                                      row, col)); 
    }  
  } 
}

template<unsigned int channels>
__global__ void shift_kernel(Pixel<channels> *d_pixels, int width, int height,
                            struct filter_args extra) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  #ifdef _DEBUG
    assert(d_pixels != nullptr);
  #endif

  #pragma unroll
  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    #pragma unroll
    for(int channel = 0; channel < channels; channel++) {
      short channel_val = d_pixels[pixel_idx]->at(channel);
      d_pixels[pixel_idx].set(channel, shift_colours(channel_val, extra, channel));
    }
  }
}

template<unsigned int channels>
__global__ void brightness_kernel(Pixel<channels> *d_pixels, int width, int height,
                                  struct filter_args extra) {
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

template<unsigned int channels>
__global__ void tint_kernel(Pixel<channels> *d_pixels, int width, int height,
                            struct filter_args extra) {
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

template<unsigned int channels>
__global__ void invert_kernel(Pixel<channels> *d_pixels, int width, int height,
                              struct filter_args extra) {
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

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int width, int height,
                          const Pixel<channels> __restrict_arr *smallest,
                          const Pixel<channels> __restrict_arr *largest,
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

// template void create_texture_object<3u>(void *h_pinned_input, int width, int height, cudaArray_t* cu_array,
//                                         cudaTextureObject_t* tex_obj);

// template void create_texture_object<4u>(void *h_pinned_input, int width, int height, cudaArray_t* cu_array,
//                                         cudaTextureObject_t* tex_obj);    

template void run_kernel(const char *filter_name, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height, struct filter_args extra);

template void run_kernel(const char *filter_name, const Pixel<4u> *input, 
                 Pixel<4u> *output, int width, int height, struct filter_args extra);

template __global__ void shift_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                         struct filter_args extra);

template __global__ void shift_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                          struct filter_args extra);

template __device__ __forceinline__ short apply_filter<3u>(const Pixel<3u> *device_input,
                                                        unsigned int mask, int width, int height, int row, int col);

template __device__ __forceinline__ short apply_filter<4u>(const Pixel<4u> *device_input, unsigned int mask,
                                                         int width, int height, int row, int col);

template __global__ void filter_kernel<3u>(const Pixel<3u> *d_in, Pixel<3u> *out, int width, int height,
                                           const struct filter_args args);

template __global__ void filter_kernel<4u>(const Pixel<4u> *d_in, Pixel<4u> *out, int width, int height,
                                           const struct filter_args args);       

template __global__ void brightness_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                               struct filter_args extra);

template __global__ void brightness_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                                struct filter_args extra);        

template __global__ void tint_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                        struct filter_args extra);  
                                      
template __global__ void tint_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                         struct filter_args extra);

template __global__ void invert_kernel<3u>(Pixel<3u> *d_pixels, int width, int height,
                                           struct filter_args extra);

template __global__ void invert_kernel<4u>(Pixel<4u> *d_pixels, int width, int height,
                                            struct filter_args extra);      