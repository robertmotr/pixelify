#ifndef __PIXEL__H
#define __PIXEL__H

#define PIXEL_NULL_CHANNEL INT_MIN

template<unsigned int channels>
struct Pixel {
    int data[channels];

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

template<unsigned int channels>
unsigned char *pixel_to_raw_image(const Pixel<channels> *input, unsigned int size) {
    unsigned char *output = new unsigned char[size * channels];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            if(input[i].data[j] < 0 || input[i].data[j] > 255) {
                printf("Pixel value out of range: %d\n", input[i].data[j]);
            }

            output[i * channels + j] = (unsigned char) input[i].data[j];
        }
    }
    return output;
}

template<unsigned int channels>
Pixel<channels> *raw_image_to_pixel(const unsigned char *input, unsigned int size) {
    Pixel<channels> *output = new Pixel<channels>[size];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < channels; j++) {
            output[i].data[j] = (int) input[i * channels + j];
        }
    }
    return output;
}

template<unsigned int channels>
void fill_pixels(Pixel<channels> *input, unsigned int size, int value) {
    if(value == PIXEL_NULL_CHANNEL) 
    // then we just randomize each channel
    {
        for (unsigned int i = 0; i < size; i++) {
            for (unsigned int j = 0; j < channels; j++) {
                input[i].data[j] = rand() % 255;
            }
        }
    }
    else {
        for (unsigned int i = 0; i < size; i++) {
            for (unsigned int j = 0; j < channels; j++) {
                input[i].data[j] = value;
            }
        }
    }
}

#endif