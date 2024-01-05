#include "gtest/gtest.h"
#include "kernel.h"
#include "cub/cub.cuh"
#include "reduce.h"

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    *os << "Pixel(" << pixel.data[0] << ", " << pixel.data[1] << ", " << pixel.data[2] << ")";
}

void print_to(const Pixel<4>& pixel, ::std::ostream* os) {
    *os << "Pixel(" << pixel.data[0] << ", " << pixel.data[1] << ", " << pixel.data[2] << pixel.data[3] << ", " << ")";
}

const int8_t IDENTITY_FILTER[] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

// const int8_t EDGE_FILTER[] = {
//     0, 1, 0,
//     1, -4, 1,
//     0, 1, 0
// };

const int8_t BOX_BLUR_FILTER[] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};

// const int8_t GAUSSIAN_BLUR_FILTER[] = {
//     1, 2, 1,
//     2, 4, 2,
//     1, 2, 1
// };

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
        {19, 20, 21}, {22, 23, 24}, {25, 26, 27}
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
    
    ASSERT_EQ(*h_largest, input[8]);
    ASSERT_EQ(*h_smallest, input[0]);

    // now test normalization
    normalize<3><<<1, 1024>>>(d_output, 3, 3, d_smallest, d_largest);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("synchronize after running normalization");

    cudaMemcpy(output, d_output, sizeof(Pixel<3>) * 9, cudaMemcpyDeviceToHost);
    Pixel<3> new_expected[9] = {
        {0, 0, 0}, {31, 31, 31}, {63, 63, 63},
        {95, 95, 95}, {127, 127, 127}, {159, 159, 159},
        {191, 191, 191}, {223, 223, 223}, {255, 255, 255}};
    
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
    Pixel<3> expected[9] = {
        {0, 0, 0}, {31, 31, 31}, {63, 63, 63},
        {95, 95, 95}, {127, 127, 127}, {159, 159, 159},
        {191, 191, 191}, {223, 223, 223}, {255, 255, 255}};

    Pixel<3> output[9] = {0};

    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 3, 3);

    // check output == expected
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}