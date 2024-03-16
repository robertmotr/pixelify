#ifndef __PIXEL__H
#define __PIXEL__H

#include <cuda_runtime.h>
#include <iostream>

// imgui requires RGBA
#define INTERNAL_CHANNEL_SIZE 4

template<unsigned int channels>
struct Pixel {
    short4 data;

    __device__ __host__ __forceinline__ void set(const unsigned int i, const short val) {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return;
            }
        #endif
        if(i == 0) {
            data.x = val;
        } 
        else if(i == 1) {
            data.y = val;
        }
        else if(i == 2) {
            data.z = val;
        }
        else {
            data.w = val;
        }
    }

    __device__ __host__ __forceinline__ const short* at_ptr(const unsigned int i) {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return nullptr;
            }
        #endif

        if(i == 0) {
            return &data.x;
        } 
        else if(i == 1) {
            return &data.y;
        }
        else if(i == 2) {
            return &data.z;
        }
        else {
            return &data.w;
        }
    }

    __device__ __host__ __forceinline__ const short* at_ptr(const unsigned int i) const {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return nullptr;
            }
        #endif

        if(i == 0) {
            return &data.x;
        } 
        else if(i == 1) {
            return &data.y;
        }
        else if(i == 2) {
            return &data.z;
        }
        else {
            return &data.w;
        }
    }

    __device__ __host__ __forceinline__ short at(const unsigned int i) {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return -1;
            }
        #endif

        if(i == 0) {
            return data.x;
        } 
        else if(i == 1) {
            return data.y;
        }
        else if(i == 2) {
            return data.z;
        }
        else {
            return data.w;
        }
    }

    __device__ __host__ __forceinline__ short at(const unsigned int i) const {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return -1;
            }
        #endif

        if(i == 0) {
            return data.x;
        } 
        else if(i == 1) {
            return data.y;
        }
        else if(i == 2) {
            return data.z;
        }
        else {
            return data.w;
        }
    }
    
    __host__ __device__
    bool operator==(const Pixel &other) const {
        return data.x == other.data.x && data.y == other.data.y && data.z == other.data.z && data.w == other.data.w;
    }

    template<typename... Args>
    __host__ __device__
    Pixel(Args... args) {
        static_assert(sizeof...(Args) <= 4, "Too many arguments for Pixel constructor");
        
        if constexpr(sizeof...(Args) >= 1) {
            data.x = std::get<0>(std::make_tuple(args...));
        }
        if constexpr(sizeof...(Args) >= 2) {
            data.y = std::get<1>(std::make_tuple(args...));
        }
        if constexpr(sizeof...(Args) >= 3) {
            data.z = std::get<2>(std::make_tuple(args...));
        }
        if constexpr(sizeof...(Args) == 4) {
            data.w = std::get<3>(std::make_tuple(args...));
        }
    }

    
    __host__ __device__ 
    Pixel(std::initializer_list<short> values) {
        auto i = values.begin();
        unsigned long size = values.size();

        if (size == 1) {
            // If there's only one value, set all channels to that value
            short singleValue = *i;
            data = make_short4(singleValue, singleValue, singleValue, (channels == 3) ? 255 : singleValue);
        }
        else if (size == channels) {
            // If there are as many values as channels, set each channel individually
            data.x = *i;
            data.y = *(++i);
            data.z = *(++i);
            data.w = (channels == 3) ? 255 : *(++i);
        } else {
            // Handle the case where the number of values doesn't match the number of channels
            // For simplicity, this example sets all channels to 0 in case of mismatch
            #ifdef _DEBUG
                printf("Number of values doesn't match the number of channels for pixel initializer list\n");
                printf("Number of values: %lu, Number of channels: %d\n", size, channels);
                printf("Error at line: %d\n", __LINE__);
            #endif
            data = make_short4(0, 0, 0, 255);
        }
    }

    __host__ __device__ 
    Pixel(short val) {
        data = make_short4(val, val, val, (channels == 3) ? 255 : val);
    }
};

template<unsigned int channels>
void imgui_get_raw_image(const Pixel<channels> *input, unsigned char *output, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i * INTERNAL_CHANNEL_SIZE + j] = static_cast<unsigned char>(input[i].at(j));
        }
        // if rgb set 4th byte to fully opaque
        if (channels == 3) {
            output[i * INTERNAL_CHANNEL_SIZE + 3] = 255;
        }
    }
}

template<unsigned int channels>
void imgui_get_pixels(const unsigned char *input, Pixel<channels> *output, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i].set(j, static_cast<short>(input[i * INTERNAL_CHANNEL_SIZE + j]));
        }
    }
}

#endif