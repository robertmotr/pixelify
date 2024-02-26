#include "gtest/gtest.h"
#include "kernel.cuh"
#include "cub/cub.cuh"
#include "reduce.cuh"
#include "filters.h"
#include "filter_impl.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const filter** filters = init_filters();

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    // recall pixels have short4
    *os << "(" << pixel.data.x << ", " << pixel.data.y << ", " << pixel.data.z << ")";
}

void print_to(const Pixel<4>& pixel, ::std::ostream* os) {
    // recall pixels have short4
    *os << "(" << pixel.data.x << ", " << pixel.data.y << ", " << pixel.data.z << ", " << pixel.data.w << ")";
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

    Pixel<3> pixels_expected[16] = {
        {0, 0, 255}, {255, 255, 255}, {0, 0, 255}, {255, 255, 255},
        {0, 0, 255}, {255, 255, 255}, {0, 0, 255}, {255, 255, 255},
        {0, 0, 255}, {255, 255, 255}, {0, 0, 255}, {255, 255, 255},
        {0, 0, 255}, {255, 255, 255}, {0, 0, 255}, {255, 255, 255}
    };

    Pixel<3> *d_pixels;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);

    struct filter_args args;
    args.alpha_shift = 0;
    args.red_shift = 0;
    args.green_shift = 0;
    args.blue_shift = 255;

    shift_kernel<3><<<1, 1024>>>(d_pixels, 4, 4, args);

    cudaMemcpy(pixels, d_pixels, 16 * sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(pixels[i], pixels_expected[i]) << "Mismatch at index " << i;
    }
}

TEST(KernelHelpers, brightness_kernel_test);
TEST(KernelHelpers, tint_kernel_test);
TEST(KernelHelpers, invert_kernel_test);
TEST(Reduction, three_channels);
TEST(Reduction, four_channels);
TEST(Reduction, mountain_with_python_script);
TEST(FilterKernel, identity_3);
TEST(FilterKernel, identity_4);
TEST(FilterKernel, identity_mountain);
TEST(FilterKernel, identity_garden);
TEST(FilterKernel, identity_helmet);
TEST(FilterKernel, identity_phone);
TEST(FilterKernel, sharpen_3);
TEST(FilterKernel, sharpen_4);
TEST(FilterKernel, blur_3);
TEST(FilterKernel, blur_4);

TEST(RunKernel, identity_3);
TEST(RunKernel, identity_4);
TEST(RunKernel, identity_mountain);
TEST(RunKernel, identity_garden);
TEST(RunKernel, identity_helmet);
TEST(RunKernel, identity_phone);

int main(int argc, char **argv) {
    setenv("current_dir", getenv("PWD"), 1);
    const char* current_dir = getenv("current_dir");
    if(current_dir != NULL) {
        // set it one layer outside i.e ../
        char *parent_dir = new char[strlen(current_dir) + 3];
        strcpy(parent_dir, current_dir);
        strcat(parent_dir, "/..");
        // now set this to current_dir
        setenv("current_dir", parent_dir, 1);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}