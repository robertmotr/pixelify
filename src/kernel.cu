#include "kernel.h"
#include "reduce.h"
#include "pixel.h"
#include "filter.h"

template<unsigned int channels>
void run_kernel(const char *filter_name, const Pixel<channels> *input,
                 Pixel<channels> *output, int width, int height, struct kernel_args extra) {

  filter *h_filter = create_filter_from_strength(filter_name, width, height, extra.filter_strength);

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
  filter *device_filter;

  // MALLOCS ON DEVICE
  cudaMalloc(&device_input, pixels * sizeof(Pixel<channels>));
  cudaMalloc(&device_output, pixels * sizeof(Pixel<channels>));
  if(filter_name != "NULL") {
    cudaMalloc(&device_filter, sizeof(filter));
    cudaMalloc(&device_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(int));
    cudaMalloc(&device_filter->filter_name, sizeof(char) * h_filter->name_size);
  }
  cudaMalloc(&d_largest, sizeof(Pixel<channels>));
  cudaMalloc(&d_smallest, sizeof(Pixel<channels>));

  // MEMCPYS FROM HOST TO DEVICE
  cudaMemcpy(device_input, h_pinned_input, pixels * sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  if(filter_name != "NULL") {
    cudaMemcpy(&device_filter->filter_dimension, &device_filter->filter_dimension, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_filter->name_size, &h_filter->name_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&device_filter->filter_name, h_filter->filter_name, sizeof(char) * h_filter->name_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_filter->filter_data, h_filter->filter_data, h_filter->filter_dimension * h_filter->filter_dimension * sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);
  cudaMemcpy(d_largest, h_largest, sizeof(Pixel<channels>), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR("cuda memcpys and mallocs");

  // apply filter first if filter is not NULL
  // then apply everything else in the kernel_args struct
  if(filter_name != "NULL") {
    kernel<channels><<<gridSize, blockSize>>>(device_filter, device_input, device_output, width, height, OP_FILTER, extra);
    CUDA_CHECK_ERROR("launching filter kernel");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("sync after filter kernel");
  }
  if(extra.alpha_shift != 0 || extra.red_shift != 0 || extra.green_shift != 0 || extra.blue_shift != 0) {
    kernel<channels><<<gridSize, blockSize>>>(NULL, device_input, device_output, width, height, OP_SHIFT_COLOURS, extra);
    CUDA_CHECK_ERROR("launching shift colour kernel");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("sync after shift colour kernel");
  } 
  if(extra.brightness != 0) {
    kernel<channels><<<gridSize, blockSize>>>(NULL, device_input, device_output, width, height, OP_BRIGHTNESS, extra);
    CUDA_CHECK_ERROR("launching brightness kernel");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("sync after brightness kernel");
  }
  if(std::any_of(std::begin(extra.tint), std::end(extra.tint), [](char i){return i != 0;})) {
    kernel<channels><<<gridSize, blockSize>>>(NULL, device_input, device_output, width, height, OP_TINT, extra);
    CUDA_CHECK_ERROR("launching tint kernel");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("sync after tint kernel");
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
  free(h_smallest); free(h_largest); free(h_filter);
  cudaFree(device_filter);
  cudaFree(d_smallest); cudaFree(d_largest);
  cudaFree(device_input); cudaFree(device_output);
}

template<unsigned int channels>
__global__  void kernel(const filter *filter, const Pixel<channels> *input, Pixel<channels> *output, 
                        int width, int height, unsigned char operation, struct kernel_args extra) {
    
    printf("----DEBUGGING STUFF----\n");
    printf("width: %d, height: %d\n", width, height);
    printf("operation: %d\n", operation);
    printf("STRUCT ARGS\n");
    printf("alpha shift: %d\n", extra.alpha_shift);
    printf("red shift: %d\n", extra.red_shift);
    printf("green shift: %d\n", extra.green_shift);
    printf("blue shift: %d\n", extra.blue_shift);
    printf("brightness: %d\n", extra.brightness);
    printf("blend factor: %f\n", extra.blend_factor);
    printf("normalize: %d\n", extra.normalize);
    printf("tint: %d, %d, %d, %d\n", extra.tint[0], extra.tint[1], extra.tint[2], extra.tint[3]);
    printf("----END DEBUGGING STUFF----\n");

    // print filter data
    if(filter != NULL) {
      printf("----FILTER DATA----\n");
      printf("filter dim: %d\n", filter->filter_dimension);
      printf("filter data: \n");
      for(int i = 0; i < filter->filter_dimension; i++) {
        for(int j = 0; j < filter->filter_dimension; j++) {
          printf("%d ", filter->filter_data[i * filter->filter_dimension + j]);
        }
        printf("\n");
      }
      printf("----END FILTER DATA----\n");
    }
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    #pragma unroll // unroll loop for performance
    for(int channel = 0; channel < channels; channel++) {
      if(operation == OP_FILTER) {
        int sum = apply_filter<channels>(input, filter, channel, width, height, row, col);
        output[pixel_idx].data[channel] = sum;
      } else if(operation == OP_SHIFT_COLOURS) {
        output[pixel_idx].data[channel] = shift_colours(input[pixel_idx].data[channel], extra, channel);
      } else if(operation == OP_BRIGHTNESS) {
        output[pixel_idx].data[channel] = input[pixel_idx].data[channel] * (100 + extra.brightness) / 100;
      }
      else if(operation == OP_TINT) {
        output[pixel_idx].data[channel] = (1 - (float)(extra.blend_factor / 100)) * extra.tint[channel] + 
                                          (float)(extra.blend_factor / 100) * input[pixel_idx].data[channel];
      }
    }
  }
}

template<unsigned int channels>
__global__ void normalize(Pixel<channels> *target, int width, int height,
                           const Pixel<channels> *smallest, const Pixel<channels> *largest, bool normalize_or_clamp) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int total_threads = blockDim.x * gridDim.x;

  for(int pixel_idx = tid; pixel_idx < width * height; pixel_idx += total_threads) {
    if(normalize_or_clamp) {
      normalize_pixel<channels>(target, pixel_idx, smallest, largest);
    } else {
      clamp_pixels<channels>(target, pixel_idx);
    }
  }
}

// explicitly instantiate
template void run_kernel<3u>(const char* filter_name, const Pixel<3u> *input,
                 Pixel<3u> *output, int width, int height, struct kernel_args extra);

template void run_kernel<4u>(const char* filter_name, const Pixel<4u> *input,
                  Pixel<4u> *output, int width, int height, struct kernel_args extra);

template __global__ void kernel<3u>(const filter *filter, const Pixel<3u> *input, 
                                  Pixel<3u> *output, int width, int height,
                                  unsigned char operation, struct kernel_args extra);

template __global__ void kernel<4u>(const filter *filter, const Pixel<4u> *input, 
                                  Pixel<4u> *output, int width, int height,
                                  unsigned char operation, struct kernel_args extra);

template __global__ void normalize<3u>(Pixel<3u> *target, int width, int height,
                           const Pixel<3u> *smallest, const Pixel<3u> *largest, bool normalize_or_clamp);

template __global__ void normalize<4u>(Pixel<4u> *target, int width, int height,
                            const Pixel<4u> *smallest, const Pixel<4u> *largest, bool normalize_or_clamp);

template void image_reduction<3u>(const Pixel<3u> *d_input, Pixel<3u>* d_result, unsigned int size, bool op);

template void image_reduction<4u>(const Pixel<4u> *d_input, Pixel<4u>* d_result, unsigned int size, bool op);