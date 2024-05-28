#include "kernel.cuh"
#include "filters.h"
#include "filter_impl.h"
#include "gtest/gtest.h"
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#pragma GCC diagnostic pop

constexpr unsigned int channels = 3; 
using PixelType = Pixel<channels>;

const filter** filters = init_filters();

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    // recall pixels have short4
    *os << "(" << pixel.data.x << ", " << pixel.data.y << ", " << pixel.data.z << ", " << pixel.data.w << ")";
}

void print_to(const Pixel<4>& pixel, ::std::ostream* os) {
    // recall pixels have short4
    *os << "(" << pixel.data.x << ", " << pixel.data.y << ", " << pixel.data.z << ", " << pixel.data.w << ")";
}

// Helper function to initialize an array of pixels
void init_image(PixelType* image, int pixels, bool random = false) {
    for (int i = 0; i < pixels; ++i) {
        for (int channel = 0; channel < channels; ++channel) {
            image[i].set(channel, random ? rand() % 256 : i % 256);
        }
    }
}

// CPU reduction function for verification
template <unsigned int channels>
Pixel<channels> cpu_image_reduction(const Pixel<channels> *image, int pixels, bool reduce_type) {
    Pixel<channels> result;
    for (int channel = 0; channel < channels; ++channel) {
        if (reduce_type == MAX_REDUCE) {
            result.set(channel, SHORT_MAX);
        } else {
            result.set(channel, SHORT_MIN);
        }
    }

    for (int i = 0; i < pixels; ++i) {
        for (int channel = 0; channel < channels; ++channel) {
            if (reduce_type == MAX_REDUCE) {
                result.set(channel, max(result.at(channel), image[i].at(channel)));
            } else {
                result.set(channel, min(result.at(channel), image[i].at(channel)));
            }
        }
    }
    return result;
}

class ImageReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
        pixels = 1024;
        image = new PixelType[pixels];
        init_image(image, pixels);

        cudaMalloc(&d_image, pixels * sizeof(PixelType));
        cudaMalloc(&d_result, sizeof(PixelType));

        cudaMemcpy(d_image, image, pixels * sizeof(PixelType), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        // Cleanup
        delete[] image;
        cudaFree(d_image);
        cudaFree(d_result);
    }

    int pixels;
    PixelType *image;
    PixelType *d_image;
    PixelType *d_result;
};

// Test for MAX_REDUCE
TEST_F(ImageReductionTest, MaxReduce) {
    image_reduction<channels>(d_image, d_result, pixels, MAX_REDUCE);

    PixelType h_result;
    cudaMemcpy(&h_result, d_result, sizeof(PixelType), cudaMemcpyDeviceToHost);

    PixelType expected = cpu_image_reduction(image, pixels, MAX_REDUCE);

    for (int channel = 0; channel < channels; ++channel) {
        EXPECT_EQ(h_result.at(channel), expected.at(channel)) << "Mismatch in channel " << channel;
    }
}

// Test for MIN_REDUCE
TEST_F(ImageReductionTest, MinReduce) {
    image_reduction<channels>(d_image, d_result, pixels, MIN_REDUCE);

    PixelType h_result;
    cudaMemcpy(&h_result, d_result, sizeof(PixelType), cudaMemcpyDeviceToHost);

    PixelType expected = cpu_image_reduction(image, pixels, MIN_REDUCE);

    for (int channel = 0; channel < channels; ++channel) {
        EXPECT_EQ(h_result.at(channel), expected.at(channel)) << "Mismatch in channel " << channel;
    }
}

TEST(KernelHelpers, find_index) {
    int index = find_index(0, 0, 3, 3);
    ASSERT_EQ(index, -1);
    index = find_index(3, 3, 0, 0);
    ASSERT_EQ(index, 0);
    index = find_index(3, 3, 1, 1);
    ASSERT_EQ(index, 4);
    index = find_index(3, 3, 2, 2);
    ASSERT_EQ(index, 8);
    index = find_index(3, 3, 2, 1);
    ASSERT_EQ(index, 7);
}

TEST(KernelHelpers, clamp_pixels) {
    Pixel<3> pixel = {-20, -230, 300};
    clamp_pixels<3>(&pixel, 0);
    ASSERT_EQ(pixel.data.x, 0);
    ASSERT_EQ(pixel.data.y, 0);
    ASSERT_EQ(pixel.data.z, 255);

    // create a list of pixels
    Pixel<3> pixels[3] = {{-20, -230, 300}, {0, 0, 0}, {255, 255, 255}};
    clamp_pixels<3>(pixels, 0);
    clamp_pixels<3>(pixels, 1);
    clamp_pixels<3>(pixels, 2);
    ASSERT_EQ(pixels[0].data.x, 0);
    ASSERT_EQ(pixels[0].data.y, 0);
    ASSERT_EQ(pixels[0].data.z, 255);
    ASSERT_EQ(pixels[1].data.x, 0);
    ASSERT_EQ(pixels[1].data.y, 0);
    ASSERT_EQ(pixels[1].data.z, 0);
    ASSERT_EQ(pixels[2].data.x, 255);
    ASSERT_EQ(pixels[2].data.y, 255);
    ASSERT_EQ(pixels[2].data.z, 255);
}

TEST(KernelHelpers, shift_colours) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    // blue values get multiplied by two
    Pixel<3> expected[16] = {
        {0, 0, 0}, {255, 255, 510}, {0, 0, 0}, {255, 255, 510},
        {0, 0, 0}, {255, 255, 510}, {0, 0, 0}, {255, 255, 510},
        {0, 0, 0}, {255, 255, 510}, {0, 0, 0}, {255, 255, 510},
        {0, 0, 0}, {255, 255, 510}, {0, 0, 0}, {255, 255, 510}
    };

    Pixel<3> *d_pixels;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);

    struct filter_args args;
    args.alpha_shift = 0;
    args.red_shift = 0;
    args.green_shift = 0;
    args.blue_shift = 100;

    shift_kernel<3><<<1, 1024>>>(d_pixels, 4, 4, args);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, 16 * sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(pixels[i], expected[i]) << "Mismatch at index " << i;
    }
    cudaFree(d_pixels);
}

TEST(KernelHelpers, brightness_kernel_test) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> pixels_expected[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> *d_pixels;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);

    struct filter_args args;
    args.brightness = 0;

    brightness_kernel<3><<<1, 1024>>>(d_pixels, 4, 4, args);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, 16 * sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(pixels[i], pixels_expected[i]) << "Mismatch at index " << i;
    }
    cudaFree(d_pixels);
}

TEST(KernelHelpers, invert_kernel_test) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> pixels_expected[16] = {
        {255, 255, 255}, {0, 0, 0}, {255, 255, 255}, {0, 0, 0},
        {255, 255, 255}, {0, 0, 0}, {255, 255, 255}, {0, 0, 0},
        {255, 255, 255}, {0, 0, 0}, {255, 255, 255}, {0, 0, 0},
        {255, 255, 255}, {0, 0, 0}, {255, 255, 255}, {0, 0, 0}
    };

    Pixel<3> *d_pixels;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);

    struct filter_args args;

    invert_kernel<3><<<1, 1024>>>(d_pixels, 4, 4, args);
    cudaDeviceSynchronize();

    cudaMemcpy(pixels, d_pixels, 16 * sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(pixels[i], pixels_expected[i]) << "Mismatch at index " << i;
    }

    cudaFree(d_pixels);
}

TEST(KernelHelpers, clamp_pixels_1) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> expected[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> *h_pixels = new Pixel<3>[16];

    for(int i = 0; i < 16; i++) {
        h_pixels[i] = pixels[i];
        clamp_pixels<3>(&h_pixels[i], i);
    }

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(h_pixels[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST(ApplyFilter, apply_filter_identity_simple) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> expected[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };
    
    Pixel<3> *output = new Pixel<3>[16];

    filter_args args;
    memset(&args, 0, sizeof(filter_args));
    args.dimension = 3;

    run_kernel<3>("Identity", pixels, output, 4, 4, args);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}