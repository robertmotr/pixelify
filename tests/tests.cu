#include "gtest/gtest.h"
#include "kernel.h"
#include "cub/cub.cuh"
#include "reduce.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    *os << "Pixel(" << pixel.data[0] << ", " << pixel.data[1] << ", " << pixel.data[2] << ")";
}

const int8_t IDENTITY_FILTER[] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

const int8_t BOX_BLUR_FILTER[] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};

TEST(kernel_correctness, identity_filter) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1},
        {1, 1, 1}, {1, 1, 1}, {1, 1, 1}
    };

    Pixel<3> output[9] = {0};
    Pixel<3> expected[9] = {0};
    memcpy(expected, input, sizeof(Pixel<3>) * 9);

    Pixel<3> *d_input, *d_output;
    int8_t *d_filter;

    cudaMalloc(&d_filter, sizeof(int8_t) * 9);  
    cudaMalloc(&d_input, sizeof(Pixel<3>) * 9);
    cudaMalloc(&d_output, sizeof(Pixel<3>) * 9);

    cudaMemcpy(d_filter, IDENTITY_FILTER, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, sizeof(Pixel<3>) * 9, cudaMemcpyHostToDevice);

    kernel<3><<<1, 1024>>>(d_filter, 3, d_input, d_output, 3, 3);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after kernel");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);
    // because its identity filter output == input
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    cudaFree(d_filter);
    cudaFree(d_input);
    cudaFree(d_output);
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

    Pixel<3> *d_input, *d_output;
    int8_t *d_filter;

    cudaMalloc(&d_filter, sizeof(int8_t) * 9);
    cudaMalloc(&d_input, sizeof(Pixel<3>) * 16);
    cudaMalloc(&d_output, sizeof(Pixel<3>) * 16);

    cudaMemcpy(d_input, input, sizeof(Pixel<3>) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, IDENTITY_FILTER, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);

    kernel<3><<<1, 1024>>>(d_filter, 3, d_input, d_output, 4, 4);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after kernel");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 16, cudaMemcpyDeviceToHost);
    
    // assert expected == output
    for (int i = 0; i < 16; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
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

    Pixel<3> *d_input, *d_output;
    int8_t *d_filter;

    cudaMalloc(&d_filter, sizeof(int8_t) * 9);
    cudaMalloc(&d_input, sizeof(Pixel<3>) * 9);
    cudaMalloc(&d_output, sizeof(Pixel<3>) * 9);

    cudaMemcpy(d_filter, BOX_BLUR_FILTER, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, sizeof(Pixel<3>) * 9, cudaMemcpyHostToDevice);

    kernel<3><<<1, 1024>>>(d_filter, 3, d_input, d_output, 3, 3);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after kernel");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);

    // assert output == expected
    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

TEST(kernel_correctness, identity_filter_sample_image_with_stb) {
    int width, height, channels;

    int ok = stbi_info("/home/robert/Desktop/pixelify/sample_images/phone.png", &width, &height, &channels);
    if(ok != 1) {
        printf("Failed to get image properties: %s\n", stbi_failure_reason());
        FAIL();
    }

    // print width, height, channels etc
    printf("Width: %d\nHeight: %d\nChannels: %d\n", width, height, channels);

    unsigned char* image_data = stbi_load("/home/robert/Desktop/pixelify/sample_images/phone.png", &width, &height, &channels, 0);
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        FAIL();
    }
    Pixel<4> *pixels_in = raw_image_to_pixel<4>(image_data, width * height);
    Pixel<4> *pixels_out = new Pixel<4>[width * height];
    
    Pixel<4> *d_input, *d_output;
    int8_t *d_filter;

    cudaMalloc(&d_filter, sizeof(int8_t) * 9);
    cudaMalloc(&d_input, sizeof(Pixel<4>) * width * height);
    cudaMalloc(&d_output, sizeof(Pixel<4>) * width * height);

    cudaMemcpy(d_input, pixels_in, sizeof(Pixel<4>) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, IDENTITY_FILTER, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);

    kernel<4><<<1, 1024>>>(d_filter, 3, d_input, d_output, 32, 32);
    CUDA_CHECK_ERROR("kernel error");
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after kernel");

    cudaMemcpy(pixels_out, d_output, sizeof(Pixel<4>) * width * height, cudaMemcpyDeviceToHost);

    unsigned char *image_out = pixel_to_raw_image<4>(pixels_out, width * height);

    // assert image_data == image_out
    for(int i = 0; i < width * height * channels; i++) {
        ASSERT_EQ(image_data[i], image_out[i]) << "Mismatch at index " << i;
    }

    // cleanup
    stbi_image_free(image_data);
    delete[] pixels_in;
    delete[] pixels_out;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    SUCCEED();
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

    Pixel<3> output[9] = {0};
    Pixel<3> expected[9] = {0};
    memcpy(expected, input, sizeof(Pixel<3>) * 9);

    Pixel<3> *d_output, *d_input;
    int8_t *d_filter;
    cudaMalloc(&d_output, sizeof(Pixel<3>) * 9);
    cudaMalloc(&d_input, sizeof(Pixel<3>) * 9);
    cudaMalloc(&d_filter, sizeof(int8_t) * 9);
    cudaMemcpy(d_input, input, sizeof(Pixel<3>) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, IDENTITY_FILTER, sizeof(int8_t) * 9, cudaMemcpyHostToDevice);

    kernel<3><<<1, 1024>>>(d_filter, 3, d_input, d_output, 3, 3);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after running kernel");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);
    // assert expected == output
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    // now find max and min of output
    Pixel<3> *h_largest = new Pixel<3>{INT_MIN};
    Pixel<3> *h_smallest = new Pixel<3>{INT_MAX};

    Pixel<3> *d_largest, *d_smallest;
    cudaMalloc(&d_largest, sizeof(Pixel<3>));
    cudaMalloc(&d_smallest, sizeof(Pixel<3>));

    image_reduction<3>(d_output, d_largest, 9, MAX_REDUCE);
    image_reduction<3>(d_output, d_smallest, 9, MIN_REDUCE);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after running reduction");

    cudaMemcpy(h_largest, d_largest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_smallest, d_smallest, sizeof(Pixel<3>), cudaMemcpyDeviceToHost);
    
    ASSERT_EQ(*h_largest, input[7]);
    ASSERT_EQ(*h_smallest, input[0]);

    // now test normalization
    normalize<3><<<1, 1024>>>(d_output, 3, 3, d_smallest, d_largest);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after running normalization");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);
    Pixel<3> new_expected[9] = {
        {0, 0, 0}, {2, 2, 2}, {5, 4, 4},
        {7, 7, 6}, {10, 9, 8}, {13, 11, 10},
        {15, 14, 12}, {255, 255, 255}, {21, 19, 16}
    };
    
    // check output == new_expected
    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(new_expected[i], output[i]);
    } 

    delete h_largest;
    delete h_smallest;
    cudaFree(d_largest);
    cudaFree(d_smallest);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

TEST(image_processing_correctness, simple_case) {
    Pixel<3> input[9] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
        {10, 11, 12}, {13, 14, 15}, {16, 17, 18},
        {19, 20, 21}, {22, 23, 24}, {25, 26, 27}
    };
    Pixel<3> expected[9] = {};
    memcpy(expected, input, sizeof(Pixel<3>) * 9);

    Pixel<3> output[9] = {0};

    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 3, 3);

    // check output == expected
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
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

    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 3, 3);

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

    run_kernel<3>(IDENTITY_FILTER, 3, pixels_in, pixels_out, width, height);
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

    run_kernel<4>(IDENTITY_FILTER, 3, pixels_in, pixels_out, width, height);
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

    run_kernel<4>(IDENTITY_FILTER, 3, pixels_in, pixels_out, width, height);
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