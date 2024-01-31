#ifndef __PIXEL__H
#define __PIXEL__H

#include <iostream>

// imgui requires RGBA
#define INTERNAL_CHANNEL_SIZE 4

template<unsigned int channels>
struct Pixel {
    int data[channels];

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
    bool operator==(const Pixel<channels> &other) const {
        for (unsigned int i = 0; i < channels; i++) {
            if (data[i] != other.data[i]) {
                return false;
            }
        }
        return true;
    }

    template<typename... Args>
    __host__ __device__
    Pixel(Args... args) : data{static_cast<int>(args)...} {}

    __host__ __device__
    Pixel() : Pixel(0) {}

    __host__ __device__
    Pixel(int value) : Pixel(value, value, value) {}
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
Pixel<channels>* raw_image_to_pixel(const unsigned char *input, Pixel<channels> *output, unsigned int size) {
    Pixel<channels> *output = new Pixel<channels>[size];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i].data[j] = static_cast<int>(input[i * channels + j]);
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
            output[i].data[j] = static_cast<int>(input[i * INTERNAL_CHANNEL_SIZE + j]);
        }
    }
}

#endif