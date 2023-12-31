#include "reduce.h"
#include "kernel.h"

// explicit instantiation
template __global__ void reduce_image<3u>(const Pixel<3u> *pixels_in, Pixel<3u> *d_result,
                                          unsigned int n, bool op);

template __global__ void reduce_image<4u>(const Pixel<4u> *pixels_in, Pixel<4u> *d_result,
                                            unsigned int n, bool op);         

template<unsigned int channels>
__global__ void reduce_image(const Pixel<channels> *pixels_in, Pixel<channels> *d_result,
                            unsigned int n, bool op) {

    __shared__ Pixel<channels> result[1];
    // set result to NULL for comparison                            
    for(int ch = 0; ch < channels; ch++) {
        result->data[ch] = PIXEL_NULL_CHANNEL;
    }

    // use find_min instead of builtin min() because we use INT_MIN
    // as NULL
    int (*reduce_op)(int, int) = (op == PIXEL_MAX) ? static_cast<int(*)(int, int)>(max) : find_min;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        for(int ch = 0; ch < channels; ch++) {
            result->data[ch] = reduce_op(result->data[ch], pixels_in[i].data[ch]);
        }
    }
    
    for(int ch = 0; ch < channels; ch++) {
        result->data[ch] = block_reduce(result->data[ch], op);
    }


    if(threadIdx.x == 0) {
        for(int ch = 0; ch < channels; ch++) {
            if(op == PIXEL_MAX) {
                atomicMax(&d_result->data[ch], result->data[ch]);
            } else {
                atomicMin(&d_result->data[ch], result->data[ch]);
            }
        }
    }
}