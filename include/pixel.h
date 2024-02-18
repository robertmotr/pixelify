#ifndef __PIXEL__H
#define __PIXEL__H

#include <cuda_runtime.h>
#include <iostream>

// imgui requires RGBA
#define INTERNAL_CHANNEL_SIZE 4

template<unsigned int channels>
struct Pixel {
    short4 data;

    __device__ __host__ __forceinline__ void set(const unsigned int i, const short& val) {
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

    __device__ __host__ __forceinline__ short* at_ptr(const unsigned int i) {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds\n");
                return -1;
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

    __device__ __host__ __forceinline__ short* at_ptr(const unsigned int i) const {
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

    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "Pixel(";
        for (int i = 0; i < channels; ++i) {
            // Access components using .x, .y, .z, .w
            if (i == 0) os << pixel.data.x;
            else if (i == 1) os << ", " << pixel.data.y;
            else if (i == 2) os << ", " << pixel.data.z;
            else if (i == 3) os << ", " << pixel.data.w;

            if (i < channels - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }

    __host__ __device__
    bool operator==(const Pixel &other) const {
        return data.x == other.data.x && data.y == other.data.y && data.z == other.data.z && data.w == other.data.w;
    }

    // Constructor with variadic template
    template<typename... Args>
    __host__ __device__
    Pixel(Args... args) : data{static_cast<short>(args)...} {
        static_assert(sizeof...(Args) <= 4, "Too many arguments for Pixel constructor");
        if constexpr(sizeof...(Args) == 3) {
            // Set the 4th byte to 255
            data.w = 255;
        }
    }

    // Constructor with initializer_list
    __host__ __device__
    Pixel(std::initializer_list<short> values) {
        // if values == 1 then set all channels to that value
        // iff channels == 4, otherwise set 4th byte to 255
        if (values.size() == 1) {
            auto value = *values.begin();
            data = make_short4(value, value, value, (channels == 3) ? 255 : value);
        } else {
            auto it = values.begin();
            data.x = *it++;
            data.y = (it != values.end()) ? *it++ : 0;
            data.z = (it != values.end()) ? *it++ : 0;
            data.w = (it != values.end()) ? *it : (channels == 3) ? 255 : 0;
        }
    }

    __host__ __device__ 
    Pixel(short val) {
        data = make_short4(val, val, val, (channels == 3) ? 255 : val);
    }
};

// use these two below instead
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