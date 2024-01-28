#include "gtest/gtest.h"
#include "kernel.h"
#include "cub/cub.cuh"
#include "reduce.h"
#include "filter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    *os << "Pixel(" << pixel.data[0] << ", " << pixel.data[1] << ", " << pixel.data[2] << ")";
}

TEST(kernel_correctness, identity_filter) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1}
    };
    unsigned char *image_in = pixel_to_raw_image<3>(input, 9);
    unsigned char *image_expected = pixel_to_raw_image<3>(input, 9);
    unsigned char *image_out = new unsigned char[9 * 3];

    Pixel<3> output[9] = {0};
    Pixel<3> expected[9] = {0};
    memcpy(expected, input, sizeof(Pixel<3>) * 9);

    struct kernel_args extra;

    extra.filter_strength = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;

    run_kernel<3>("Identity", input, output, 3, 3, extra);

    // assert expected == output
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    image_out = pixel_to_raw_image<3>(output, 9);

    // assert image_out == image_expected
    for (int i = 0; i < 9 * 3; i++) {
        ASSERT_EQ(image_expected[i], image_out[i]);
    }

    free(image_in);
    free(image_out);
    free(image_expected);
}

TEST(kernel_correctness, identity_filter_channels) {
    Pixel<3> input[16] = {
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12},
    {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24},
    {25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36},
    {37, 38, 39}, {40, 41, 42}, {43, 44, 45}, {46, 47, 48}
    };

    Pixel<3> output[16] = {0};

    Pixel<3> expected[16] = {0};
    memcpy(expected, input, sizeof(Pixel<3>) * 16);

    struct kernel_args extra;

    extra.filter_strength = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.red_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = false;

    run_kernel<3>("Identity", input, output, 4, 4, extra);

    // assert output == expected
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }
}

TEST(kernel_correctness, simple_box_blur) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3},
        {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
        {7, 7, 7}, {8, 8, 8}, {9, 9, 9}
    };
    Pixel<3> output[9] = {0};

    Pixel<3> expected[9] = {
        {12, 12, 12}, {21, 21, 21}, {16, 16, 16},
        {27, 27, 27}, {45, 45, 45}, {33, 33, 33},
        {24, 24, 24}, {39, 39, 39}, {28, 28, 28}
    };

    struct kernel_args extra;

    extra.filter_strength = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.red_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = false;

    run_kernel<3>("Box Blur", input, output, 3, 3, extra);

    // assert output == expected
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }
}

TEST(parallel_reduction_correctness, real_sample_image) {
    int width, height, channels;

    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    unsigned char *image_output = new unsigned char[width * height * channels];
    Pixel<3> *pixels_in = raw_image_to_pixel<3>(image_data, width * height);
    
    Pixel<3> *d_input, *d_smallest, *d_largest;

    cudaMalloc(&d_input, sizeof(Pixel<3>) * width * height);
    cudaMalloc(&d_smallest, sizeof(Pixel<3>));
    cudaMalloc(&d_largest, sizeof(Pixel<3>));

    cudaMemcpy(d_input, pixels_in, sizeof(Pixel<3>) * width * height, cudaMemcpyHostToDevice);

    image_reduction<3>(d_input, d_largest, width * height, MAX_REDUCE);
    image_reduction<3>(d_input, d_smallest, width * height, MIN_REDUCE);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("sync after reduction");

    Pixel<3> *h_smallest = new Pixel<3>();
    Pixel<3> *h_largest = new Pixel<3>();

    cudaMemcpy(h_smallest, d_smallest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_largest, d_largest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);

    // print values of h smallest/hlargest
    char buf[100];
    sprintf(buf, "Global max values: [%d, %d, %d]\nGlobal min values: [%d, %d, %d]\n", 
            h_largest->data[0], h_largest->data[1], h_largest->data[2],
            h_smallest->data[0], h_smallest->data[1], h_smallest->data[2]);
    
    system("python3 /home/robert/Desktop/pixelify/test_reduction.py > output.txt");

    FILE *f = fopen("output.txt", "r");
    if (f == NULL) {
        perror("Error opening output.txt");
        FAIL();
    }

    char output_buf[200];
    size_t bytes_read = fread(output_buf, 1, sizeof(output_buf) - 1, f);
    output_buf[bytes_read] = '\0'; 

    fclose(f);

    printf("\nBUF RESULT:\n%s", buf);
    printf("\nOUTPUTBUF RESULT:\n%s", output_buf);

    if(strcmp(buf, output_buf) != 0) {
        std::cout << strcmp(buf, output_buf) << " STRCMP RESULT" << "\n";
        system("rm output.txt");
        FAIL();
    }
    system("rm output.txt");
    
    SUCCEED();
}

TEST(parallel_reduction_correctness, small_image) {
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

    Pixel<3> *h_largest = new Pixel<3>{INT_MIN};
    Pixel<3> *h_smallest = new Pixel<3>{INT_MAX};

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

TEST(normalization_correctness, identity_filter) {
    Pixel<3> input[9] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18},
        {19, 20, 21}, {289, 324, 367}, {25, 26, 27}
    };
    Pixel<3> output[9]; 
    
    struct kernel_args extra;

    extra.filter_strength = 0;
    extra.red_shift = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = true;

    run_kernel<3>("Identity", input, output, 3, 3, extra);

    Pixel<3> new_expected[9] = {
        {0, 0, 0}, {2, 2, 2}, {5, 4, 4},
        {7, 7, 6}, {10, 9, 8}, {13, 11, 10},
        {15, 14, 12}, {255, 255, 255}, {21, 19, 16}
    };
    
    // check output == new_expected
    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(new_expected[i], output[i]);
    } 
}

TEST(image_processing_correctness, simple_case_w_normalization) {
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

    struct kernel_args extra;

    extra.filter_strength = 0;
    extra.red_shift = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = true;

    run_kernel<3>("Identity", input, output, 3, 3, extra);

    // check output == expected
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }
}

TEST(image_processing_correctness, stb_conversion) {
    int width, height, channels;

    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    Pixel<3> *pixels_in = raw_image_to_pixel<3>(image_data, width * height);

    for(int i = 0; i < width * height; i++) {
        pixels_in[i].data[0] /= 2;
        pixels_in[i].data[1] /= 2;
        pixels_in[i].data[2] /= 2;
    }

    unsigned char *image_out = pixel_to_raw_image<3>(pixels_in, width * height);

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i] / 2, image_out[i]) << "Mismatch at index " << i;
    }

    // write image_out to file
    stbi_write_png("output.png", width, height, channels, image_out, width * channels);

    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    SUCCEED();
}

TEST(image_processing_correctness, identity_filter) {
    int width, height, channels;
    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }
    // print image properties
    printf("Image width: %d\nImage height: %d\nImage channels: %d\n", width, height, channels);

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/Puzzle_Mountain.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    Pixel<3> *pixels_in = raw_image_to_pixel<3>(image_data, width * height);
    Pixel<3> *pixels_out = new Pixel<3>[width * height];

    struct kernel_args extra;
    extra.filter_strength = 0;
    extra.red_shift = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = false;

    run_kernel<3>("Identity", pixels_in, pixels_out, 3, 3, extra);
    unsigned char *image_out = pixel_to_raw_image<3>(pixels_out, width * height);

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }
    
    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    delete[] pixels_out;
    SUCCEED();
}

TEST(image_processing_correctness, identity_filter_garden) {
    int width, height, channels;

    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/garden.png", &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    // print width, height, channels etc
    printf("Width: %d\nHeight: %d\nChannels: %d\n", width, height, channels);

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/garden.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    Pixel<4> *pixels_in = raw_image_to_pixel<4>(image_data, width * height);
    Pixel<4> *pixels_out = new Pixel<4>[width * height];

    struct kernel_args extra;
    extra.filter_strength = 0;
    extra.red_shift = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = false;

    run_kernel<4>("Identity", pixels_in, pixels_out, 3, 3, extra);
    unsigned char *image_out = pixel_to_raw_image<4>(pixels_out, width * height);

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }

    // write image_out to file
    // stbi_write_png("output.png", width, height, channels, image_out, width * channels);

    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    delete[] pixels_out;
    SUCCEED();
}

TEST(image_processing_correctness, identity_filter_helmet) {
    int width, height, channels;
    int old_w, old_h, old_c;

    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/helmet.png", &old_w, &old_h, &old_c);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    // print width, height, channels etc
    printf("Width: %d\nHeight: %d\nChannels: %d\n", old_w, old_h, old_c);

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/helmet.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }

    Pixel<4> *pixels_in = raw_image_to_pixel<4>(image_data, width * height);
    Pixel<4> *pixels_out = new Pixel<4>[width * height];

    struct kernel_args extra;
    extra.filter_strength = 0;
    extra.red_shift = 0;
    extra.green_shift = 0;
    extra.blue_shift = 0;
    extra.alpha_shift = 0;
    extra.brightness = 0;
    extra.blend_factor = 0.0f;
    extra.tint[0] = 0;
    extra.tint[1] = 0;
    extra.tint[2] = 0;
    extra.tint[3] = 0;
    extra.normalize = false;

    run_kernel<4>("Identity", pixels_in, pixels_out, 3, 3, extra);
    unsigned char *image_out = pixel_to_raw_image<4>(pixels_out, width * height);

    // assert that image is the same between image out and image data
    for (int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }

    // write image_out to file
    // stbi_write_png("output.png", width, height, channels, image_out, width * channels);

    // free memory
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] image_out;
    delete[] pixels_out;
    SUCCEED();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}