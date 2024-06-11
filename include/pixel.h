#ifndef __PIXEL__H
#define __PIXEL__H

#include <cuda_runtime.h>
#include <iostream>

// imgui requires RGBA
#define INTERNAL_CHANNEL_SIZE 4

template<unsigned int channels>
class Pixel {
public:
    short4 data;

    // Sets the value of the pixel at index i to val 
    __device__ __host__ __forceinline__
    void set(const unsigned int i, const short val) {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds, i is %d, channels is %d\n", i, channels);
                printf("at line %d\n", __LINE__);
                return;
            }
        #endif
        switch (i) {
            case 0:
                data.x = val;
                break;
            case 1:
                data.y = val;
                break;
            case 2:
                data.z = val;
                break;
            case 3:
                data.w = val;
                break;
            default:
                break;
        }
    }

    // Sets the value of the pixel at index i to val (volatile version)
    __device__ __host__ __forceinline__
    void set(const unsigned int i, const short val) volatile {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds, i is %d, channels is %d\n", i, channels);
                printf("at line %d\n", __LINE__);
                return;
            }
        #endif
        switch (i) {
            case 0:
                data.x = val;
                break;
            case 1:
                data.y = val;
                break;
            case 2:
                data.z = val;
                break;
            case 3:
                data.w = val;
                break;
            default:
                break;
        }
    }

    // Returns a pointer to the pixel at index i
    __device__ __host__ __forceinline__ 
    const short* at_ptr(const unsigned int i) const {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds, i is %d, channels is %d\n", i, channels);
                printf("at line %d\n", __LINE__);
                return nullptr;
            }
        #endif
        switch (i) {
            case 0:
                return &data.x;
            case 1:
                return &data.y;
            case 2:
                return &data.z;
            case 3:
                return &data.w;
            default:
                return nullptr;
        }
    }

    // Returns the value of the pixel at index i (const version)
    __device__ __host__ __forceinline__ 
    short at(const unsigned int i) const {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds, i is %d, channels is %d\n", i, channels);
                printf("at line %d\n", __LINE__);
                return -1;
            }
        #endif
        switch (i) {
            case 0:
                return data.x;
            case 1:
                return data.y;
            case 2:
                return data.z;
            case 3:
                return data.w;
            default:
                return -1;
        }
    }

    // Returns the value of the pixel at index i (volatile version)
    __device__ __host__ __forceinline__ 
    short at(const unsigned int i) const volatile {
        #ifdef _DEBUG
            if (i >= channels) {
                printf("index out of bounds, i is %d, channels is %d\n", i, channels);
                printf("at line %d\n", __LINE__);
                return -1;
            }
        #endif
        switch (i) {
            case 0:
                return data.x;
            case 1:
                return data.y;
            case 2:
                return data.z;
            case 3:
                return data.w;
            default:
                return -1;
        }
    }
    
    // Operator overload for comparing equality
    __host__ __device__
    bool operator==(const Pixel &other) const {
        return data.x == other.data.x && data.y == other.data.y && data.z == other.data.z && data.w == other.data.w;
    }

    // Operator overload for comparing inequality
    __host__ __device__
    bool operator!=(const Pixel &other) const {
        return !(*this == other);
    }

    // Operator overload for << to print the pixel
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.data.x << ", " << pixel.data.y << ", " << pixel.data.z << ", " << pixel.data.w << ")";
        return os;
    }

    // Operator overload for << to print the pixel (ptr version)
    friend std::ostream& operator<<(std::ostream& os, const Pixel *pixel) {
        os << "(" << pixel->data.x << ", " << pixel->data.y << ", " << pixel->data.z << ", " << pixel->data.w << ")";
        return os;
    }

    // Constructor for variadic arguments
    template<typename... Args>
    __host__ __device__ Pixel(Args... args) {
        static_assert(sizeof...(Args) <= 4, "Too many arguments for Pixel constructor");

        if constexpr (sizeof...(Args) > 0) {
            short vals[] = { static_cast<short>(args)... };
            int num_args = sizeof...(Args);

            data.x = (num_args > 0) ? vals[0] : 0;
            data.y = (num_args > 1) ? vals[1] : 0;
            data.z = (num_args > 2) ? vals[2] : 0;
            data.w = (channels == 3) ? 0 : ((num_args > 3) ? vals[3] : 0);
        } else {
            // Handle the zero-arguments case
            data.x = 0;
            data.y = 0;
            data.z = 0;
            data.w = (channels == 3) ? 0 : 255;
        }
    }

    // Constructor for initializer list
    __host__ __device__ Pixel(std::initializer_list<short> values) {
        auto i = values.begin();
        unsigned long size = values.size();

        data.x = (size > 0) ? *i : 0;
        data.y = (size > 1) ? *(i + 1) : 0;
        data.z = (size > 2) ? *(i + 2) : 0;
        data.w = (channels == 3) ? 0 : ((size > 3) ? *(i + 3) : 0);
    }

    // Default constructor
    __host__ __device__ 
    Pixel() : data({0, 0, 0, (channels == 3) ? 0 : 255}) {}

    // Constructor for single value
    __host__ __device__ 
    Pixel(short val) {
        data = make_short4(val, val, val, (channels == 3) ? 0 : val);
    }
};

// Sets output to the value of input which is a Pixel array of size size
template<unsigned int channels>
void imgui_get_raw_image(const Pixel<channels> *input, unsigned char *output, unsigned int size) {
    assert(channels <= 4);
    assert(input != nullptr);
    assert(output != nullptr);

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

// Sets output to the value of input which is a primitive RGBA array of size size
template<unsigned int channels>
void imgui_get_pixels(const unsigned char *input, Pixel<channels> *output, unsigned int size) {
    assert(channels <= 4);
    assert(input != nullptr);
    assert(output != nullptr);
    
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i].set(j, static_cast<short>(input[i * INTERNAL_CHANNEL_SIZE + j]));
        }
    }
}

#endif