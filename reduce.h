#ifndef __REDUCE__H
#define __REDUCE__H

#include "pixel.h"
#include "cub/cub.cuh"

#define MAX_REDUCE true
#define MIN_REDUCE false

template<unsigned int channels>
struct min_op {
    __device__ CUB_RUNTIME_FUNCTION __forceinline__ 
    Pixel<channels> operator()(const Pixel<channels>& a, const Pixel<channels>& b) const {
        Pixel<channels> result;
        for (int i = 0; i < channels; ++i) {
            result.data[i] = min(a.data[i], b.data[i]);
        }
        return result;
    }
};

template<unsigned int channels>
struct max_op {
    __device__ CUB_RUNTIME_FUNCTION __forceinline__
    Pixel<channels> operator()(const Pixel<channels>& a, const Pixel<channels>& b) const {
        Pixel<channels> result;
        for (int i = 0; i < channels; ++i) {
            result.data[i] = max(a.data[i], b.data[i]);
        }
        return result;
    }
};

template<unsigned int channels>
void image_reduction(Pixel<channels> *d_input, Pixel<channels> *d_result, unsigned int size, bool operation) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    if(operation == MAX_REDUCE) {
        cub::DeviceReduce::Reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_result,
                                  size,
                                  max_op<channels>(),
                                  Pixel<channels>(INT_MIN));;
        
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceReduce::Reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_result,
                                  size,
                                  max_op<channels>(),
                                  Pixel<channels>(INT_MIN));
    }
    else {
        cub::DeviceReduce::Reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_result,
                                  size,
                                  min_op<channels>(),
                                  Pixel<channels>(INT_MAX));;
        
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cub::DeviceReduce::Reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  d_input,
                                  d_result,
                                  size,
                                  min_op<channels>(),
                                  Pixel<channels>(INT_MAX));
    }

    cudaFree(d_temp_storage);
}


#endif // __REDUCE__H