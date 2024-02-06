#ifndef __PIXEL__H
#define __PIXEL__H

#include <iostream>

// imgui requires RGBA
#define INTERNAL_CHANNEL_SIZE 4

template<unsigned int channels>
struct Pixel {
    short data[4];

    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "Pixel(";
        for (int i = 0; i < channels; ++i) {
            os << pixel.data[i];
            if (i < channels - 1) {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }    

    __host__ __device__
    bool operator==(const Pixel &other) const {
        for (unsigned int i = 0; i < channels; i++) {
            if (data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }

    // if channels == 3 set 4th byte to 255
    // in constructor
    template<typename... Args>
    __host__ __device__
    Pixel(Args... args) : data{static_cast<short>(args)...} {
        static_assert(sizeof...(Args) <= 4, "Too many arguments for Pixel constructor");
        if constexpr(sizeof...(Args) == 3) {
            // Set the 4th byte to 255
            data[3] = 255;
        }
    }
    __host__ __device__
    Pixel(std::initializer_list<short> values) {
        // if values == 1 then set all channels to that value
        // iff channels == 4, otherwise set 4th byte to 255
        if (values.size() == 1) {
            for (unsigned int i = 0; i < channels; i++) {
                data[i] = *values.begin();
            }
            if constexpr(channels == 3) {
                // Set the 4th byte to 255
                data[3] = 255;
            }
        } else {
            unsigned int i = 0;
            for (auto it = values.begin(); it != values.end(); it++) {
                data[i] = *it;
                i++;
            }
        }
    }

    __host__ __device__ 
    Pixel(short val) {
        for (unsigned int i = 0; i < channels; i++) {
            data[i] = val;
        }
        if constexpr(channels == 3) {
            // Set the 4th byte to 255
            data[3] = 255;
        }
    }

    __host__ __device__
    Pixel() : Pixel(0) {}
};

// dont use this doesnt work with imgui, kept this so tests are stable
template<unsigned int channels>
unsigned char* pixel_to_raw_image(const Pixel<channels> *input, unsigned int size) {
    unsigned char *output = new unsigned char[channels * size];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i * channels + j] = static_cast<unsigned char>(input[i].data[j]);
        }
    }
    return output;
}

// dont use this doesnt work with imgui, kept this so tests are stable
template<unsigned int channels>
Pixel<channels>* raw_image_to_pixel(const unsigned char *input, unsigned int size) {
    Pixel<channels> *output = new Pixel<channels>[size];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i].data[j] = static_cast<short>(input[i * channels + j]);
        }
    }
    return output;
}

// use these two below instead
template<unsigned int channels>
void imgui_get_raw_image(const Pixel<channels> *input, unsigned char *output, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i * INTERNAL_CHANNEL_SIZE + j] = static_cast<unsigned char>(input[i].data[j]);
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
            output[i].data[j] = static_cast<short>(input[i * INTERNAL_CHANNEL_SIZE + j]);
        }
    }
}

#endif