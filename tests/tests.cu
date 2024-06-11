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
template<unsigned int channels>
void init_image(Pixel<channels> *h_image, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        for (int channel = 0; channel < channels; ++channel) {
            h_image[i].set(channel, rand() % 256);
        }
    }
}

// CPU reduction function for verification
template <unsigned int channels>
Pixel<channels> cpu_image_reduction(const Pixel<channels> *image, int pixels, bool reduce_type) {
    Pixel<channels> result;
    for (int channel = 0; channel < channels; ++channel) {
        if (reduce_type == MAX_REDUCE) {
            result.set(channel, SHORT_MIN);
        } else {
            result.set(channel, SHORT_MAX);
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
        clamp_pixels<3>(h_pixels, i);
    }

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(h_pixels[i], expected[i]) << "Mismatch at index " << i;
    }

    delete[] h_pixels;
}

TEST(OtherKernels, image_reduction_simple) {
    Pixel<3> pixels[16] = {
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255},
        {0, 0, 0}, {255, 255, 255}, {0, 0, 0}, {255, 255, 255}
    };

    Pixel<3> *d_pixels = nullptr;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    CUDA_CHECK_ERROR("malloc");
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("memcpy");

    Pixel<3> *d_expected_max, *d_expected_min;
    Pixel<3> *h_expected_max, *h_expected_min;
    cudaMalloc(&d_expected_max, sizeof(Pixel<3>));
    cudaMalloc(&d_expected_min, sizeof(Pixel<3>));

    // assert cpu image reduction is correct
    Pixel<3> expected_max = cpu_image_reduction<3>(pixels, 16, MAX_REDUCE);
    Pixel<3> expected_min = cpu_image_reduction<3>(pixels, 16, MIN_REDUCE);

    image_reduction<3>(d_pixels, d_expected_max, 16, MAX_REDUCE);
    image_reduction<3>(d_pixels, d_expected_min, 16, MIN_REDUCE);

    h_expected_max = new Pixel<3>;
    h_expected_min = new Pixel<3>;

    cudaMemcpy(h_expected_max, d_expected_max, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expected_min, d_expected_min, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(h_expected_max->at(i), 255) << "Mismatch at channel " << i;
        ASSERT_EQ(h_expected_min->at(i), 0) << "Mismatch at channel " << i;
    }

    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(h_expected_max->at(i), expected_max.at(i)) << "Mismatch at channel " << i;
        ASSERT_EQ(h_expected_min->at(i), expected_min.at(i)) << "Mismatch at channel " << i;
    }

    cudaFree(d_pixels);
    cudaFree(d_expected_max);
    cudaFree(d_expected_min);
    delete h_expected_max;
    delete h_expected_min;
}

TEST(OtherKernels, image_reduction_randomized) {
    Pixel<3> pixels[16];
    init_image<3>(pixels, 16);

    // print the values of pixels after randomization nicely
    for(int i = 0; i < 16; i++) {
        std::cout << "Pixel " << i << ": ";
        print_to(pixels[i], &std::cout);
        std::cout << std::endl;
    }

    Pixel<3> *d_pixels = nullptr;
    cudaMalloc(&d_pixels, 16 * sizeof(Pixel<3>));
    CUDA_CHECK_ERROR("malloc");
    cudaMemcpy(d_pixels, pixels, 16 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("memcpy");

    Pixel<3> *d_expected_max, *d_expected_min;
    Pixel<3> *h_expected_max, *h_expected_min;
    cudaMalloc(&d_expected_max, sizeof(Pixel<3>));
    cudaMalloc(&d_expected_min, sizeof(Pixel<3>));

    // assert cpu image reduction is correct
    Pixel<3> expected_max = cpu_image_reduction<3>(pixels, 16, MAX_REDUCE);
    Pixel<3> expected_min = cpu_image_reduction<3>(pixels, 16, MIN_REDUCE);

    image_reduction<3>(d_pixels, d_expected_max, 16, MAX_REDUCE);
    image_reduction<3>(d_pixels, d_expected_min, 16, MIN_REDUCE);

    h_expected_max = new Pixel<3>;
    h_expected_min = new Pixel<3>;

    cudaMemcpy(h_expected_max, d_expected_max, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expected_min, d_expected_min, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(h_expected_max->at(i), expected_max.at(i)) << "Mismatch at channel " << i;
        ASSERT_EQ(h_expected_min->at(i), expected_min.at(i)) << "Mismatch at channel " << i;
    }

    cudaFree(d_pixels);
    cudaFree(d_expected_max);
    cudaFree(d_expected_min);
    delete h_expected_max;
    delete h_expected_min;
}

TEST(OtherKernels, image_reduction_large_scale_randomized) {
    Pixel<3> pixels[8192];
    init_image<3>(pixels, 8192);

    Pixel<3> *d_pixels = nullptr;
    cudaMalloc(&d_pixels, 8192 * sizeof(Pixel<3>));
    CUDA_CHECK_ERROR("malloc");
    cudaMemcpy(d_pixels, pixels, 8192 * sizeof(Pixel<3>), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("memcpy");

    Pixel<3> *d_expected_max, *d_expected_min;
    Pixel<3> *h_expected_max, *h_expected_min;
    cudaMalloc(&d_expected_max, sizeof(Pixel<3>));
    cudaMalloc(&d_expected_min, sizeof(Pixel<3>));

    // assert cpu image reduction is correct
    Pixel<3> expected_max = cpu_image_reduction<3>(pixels, 8192, MAX_REDUCE);
    Pixel<3> expected_min = cpu_image_reduction<3>(pixels, 8192, MIN_REDUCE);

    image_reduction<3>(d_pixels, d_expected_max, 8192, MAX_REDUCE);
    image_reduction<3>(d_pixels, d_expected_min, 8192, MIN_REDUCE);

    h_expected_max = new Pixel<3>;
    h_expected_min = new Pixel<3>;

    cudaMemcpy(h_expected_max, d_expected_max, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_expected_min, d_expected_min, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(h_expected_max->at(i), expected_max.at(i)) << "Mismatch at channel " << i;
        ASSERT_EQ(h_expected_min->at(i), expected_min.at(i)) << "Mismatch at channel " << i;
    }

    cudaFree(d_pixels);
    cudaFree(d_expected_max);
    cudaFree(d_expected_min);
    delete h_expected_max;
    delete h_expected_min;
}

TEST(ApplyFilter, identity_randomized) {
    Pixel<3> *h_pixels = new Pixel<3>[16];
    Pixel<3> *h_output = new Pixel<3>[16];
    Pixel<3> *h_expected = new Pixel<3>[16];

    init_image<3>(h_pixels, 16);
    memcpy(h_expected, h_pixels, 16 * sizeof(Pixel<3>));

    filter_args args;
    memset(&args, 0, sizeof(filter_args));
    args.passes = 1;
    args.dimension = 3;

    run_kernel<3>("Identity", h_pixels, h_output, 4, 4, args);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(h_pixels[i], h_expected[i]) << "Mismatch at index " << i;
    }

    delete[] h_pixels;
    delete[] h_expected;
    delete[] h_output;
}

TEST(ApplyFilter, simple_box_blur) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3},
        {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
        {7, 7, 7}, {8, 8, 8}, {9, 9, 9}
    };
    Pixel<3> output[9];
    memset(output, 0, sizeof(output));

    Pixel<3> expected[9] = {
        {12, 12, 12}, {21, 21, 21}, {16, 16, 16},
        {27, 27, 27}, {45, 45, 45}, {33, 33, 33},
        {24, 24, 24}, {39, 39, 39}, {28, 28, 28}
    };

    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(input[i], input[i]) << "Mismatch at index " << i;
    }

    struct filter_args extra;
    memset(&extra, 0, sizeof(filter_args));
    extra.passes = 1;
    extra.dimension = 3;

    run_kernel<3>("Box Blur", input, output, 3, 3, extra);

    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]) << "Mismatch at index " << i;
    }
}

TEST(Reduction, small_image) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3},
        {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
        {7, 7, 7}, {8, 8, 8}, {9, 9, 9}
    };

    Pixel<3> *d_input;
    cudaMalloc(&d_input, sizeof(Pixel<3>) * 9);
    cudaMemcpy(d_input, input, sizeof(Pixel<3>) * 9, cudaMemcpyHostToDevice);

    Pixel<3> *d_largest;
    Pixel<3> *d_smallest;

    Pixel<3> *h_largest = new Pixel<3>{SHORT_MIN};
    Pixel<3> *h_smallest = new Pixel<3>{SHORT_MAX};

    cudaMalloc(&d_largest, sizeof(Pixel<3>));
    cudaMalloc(&d_smallest, sizeof(Pixel<3>));

    cudaMemcpy(d_largest, h_largest, sizeof(Pixel<3>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_smallest, h_smallest, sizeof(Pixel<3>), cudaMemcpyHostToDevice);

    image_reduction<3>(d_input, d_largest, 9, MAX_REDUCE);
    image_reduction<3>(d_input, d_smallest, 9, MIN_REDUCE);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after reduction");

    // copy back d_input to double check its unchanged
    Pixel<3> *h_input = new Pixel<3>[9];
    cudaMemcpy(h_input, d_input, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(input[i], h_input[i]);
    }

    cudaMemcpy(h_largest, d_largest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smallest, d_smallest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    ASSERT_EQ(*h_largest, input[8]);
    ASSERT_EQ(*h_smallest, input[0]);

    cudaFree(d_input);
    cudaFree(d_largest);
    cudaFree(d_smallest);
    delete h_largest;
    delete h_smallest;
}

TEST(Normalization, identity) {
    Pixel<3> input[9] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18},
        {19, 20, 21}, {289, 324, 367}, {25, 26, 27}
    };
    Pixel<3> output[9]; 

    struct filter_args extra;
    memset(&extra, 0, sizeof(filter_args));
    extra.passes = 1;
    extra.dimension = 3;
    extra.normalize = true;

    run_kernel<3>("Identity", input, output, 3, 3, extra);

    Pixel<3> expected[9] = {
        {0, 0, 0}, {2, 2, 2}, {5, 4, 4},
        {7, 7, 6}, {10, 9, 8}, {13, 11, 10},
        {15, 14, 12}, {255, 255, 255}, {21, 19, 16}
    };

    // check output == new_expected
    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]) << "Mismatch at index " << i;
    } 
}

TEST(Normalization, simple_case) {
    Pixel<3> input[9] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18},
        {19, 20, 21}, {289, 324, 367}, {25, 26, 27}
    };

    Pixel<3> output[9] = {0};
    Pixel<3> expected[9] = {
        {0, 0, 0}, {2, 2, 2}, {5, 4, 4},
        {7, 7, 6}, {10, 9, 8}, {13, 11, 10},
        {15, 14, 12}, {255, 255, 255}, {21, 19, 16}
    };

    struct filter_args extra;
    memset(&extra, 0, sizeof(filter_args));
    extra.passes = 1;
    extra.dimension = 3;
    extra.normalize = true;

    run_kernel<3>("Identity", input, output, 3, 3, extra);

    // check output == expected
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }
}

TEST(ImageProcessing, stb_conversion) {
    int width, height, channels;

    const char *env_var = getenv("current_dir");
    char *full_path = NULL;
    if(env_var != NULL) {
        full_path = new char[strlen(env_var) + strlen("/sample_images/Puzzle_Mountain.png") + 1];
        printf("Current dir: %s\nImage_processing_correctness running for stb_conversion\n", env_var);
        strcpy(full_path, env_var);
        strcat(full_path, "/sample_images/Puzzle_Mountain.png");
    }
    else {
        printf("Error: current_dir environment variable not set\n");
        FAIL();
    }

    int ok = stbi_info(full_path, &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    unsigned char* image_data = stbi_load(full_path, &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    Pixel<4> *pixels_in = new Pixel<4>[width * height];
    imgui_get_pixels<4>(image_data, pixels_in, width * height);

    if(pixels_in == nullptr) {
        printf("Failed to convert image to pixels\n");
        FAIL();
    }

    unsigned char *image_out = new unsigned char[width * height * INTERNAL_CHANNEL_SIZE];
    imgui_get_raw_image<4>(pixels_in, image_out, width * height);

    if(image_out == nullptr) {
        printf("Failed to convert image to raw image\n");
        FAIL();
    }

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }

    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    delete[] full_path;
}

TEST(ImageProcessing, identity_filter_on_real_image) {
    const char *env_var = getenv("current_dir");
    char *full_path = NULL;
    if(env_var != NULL) {
        full_path = new char[strlen(env_var) + strlen("/sample_images/Puzzle_Mountain.png") + 1];
        printf("Current dir: %s\nRunning identity_filter image processing correctness\n", env_var);
        strcpy(full_path, env_var);
        strcat(full_path, "/sample_images/Puzzle_Mountain.png");
    }
    else {
        printf("Error: current_dir environment variable not set\n");
        FAIL();
    }

    int width, height, channels;
    int ok = stbi_info(full_path, &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }
    // print image properties
    printf("Image width: %d\nImage height: %d\nImage channels: %d\n", width, height, channels);

    unsigned char* image_data = stbi_load(full_path, &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }
    Pixel<4> *pixels_in = new Pixel<4>[width * height];
    imgui_get_pixels<4>(image_data, pixels_in, width * height);
    Pixel<4> *pixels_out = new Pixel<4>[width * height];

    struct filter_args extra;
    memset(&extra, 0, sizeof(filter_args));
    extra.passes = 1;
    extra.dimension = 3;

    run_kernel<4>("Identity", pixels_in, pixels_out, width, height, extra);
    unsigned char *image_out = new unsigned char[width * height * INTERNAL_CHANNEL_SIZE];
    imgui_get_raw_image<4>(pixels_out, image_out, width * height);

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }

    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    delete[] pixels_out;
    delete[] full_path;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}