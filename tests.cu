#include "gtest/gtest.h"
#include "kernel.h"
#include "cub/cub.cuh"
#include "cub/util_debug.cuh"

void print_to(const Pixel<3>& pixel, ::std::ostream* os) {
    *os << "Pixel(" << pixel.data[0] << ", " << pixel.data[1] << ", " << pixel.data[2] << ")";
}

template<unsigned int channels>
struct min_op {
    __device__ Pixel<channels> operator()(const Pixel<channels>& a, const Pixel<channels>& b) const {
        Pixel<channels> result;
        for (int i = 0; i < channels; ++i) {
            result.data[i] = min(a.data[i], b.data[i]);
        }
        return result;
    }
};

template<unsigned int channels>
struct max_op {
    __device__ Pixel<channels> operator()(const Pixel<channels>& a, const Pixel<channels>& b) const {
        Pixel<channels> result;
        for (int i = 0; i < channels; ++i) {
            result.data[i] = max(a.data[i], b.data[i]);
        }
        return result;
    }
};


const int8_t IDENTITY_FILTER[] = {
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};

const int8_t EDGE_FILTER[] = {
    0, 1, 0,
    1, -4, 1,
    0, 1, 0
};

const int8_t BOX_BLUR_FILTER[] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};

const int8_t GAUSSIAN_BLUR_FILTER[] = {
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};

TEST(kernel_correctness, identity_filter) {
    Pixel<3> input[9] = {0};
    fill_pixels(input, 9, 1); // fill with 1s
    Pixel<3> output[9] = {0};
    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 3, 3);

    // because its identity filter output == input
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(input[i], output[i]);
    }
}

TEST(kernel_correctness, identity_filter_random_image) {
    Pixel<3> input[9] = {0};
    fill_pixels(input, 9, PIXEL_NULL_CHANNEL); // fill with random
    Pixel<3> output[9] = {0};
    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 3, 3);

    // because its identity filter output == input
    for (int i = 0; i < 9; i++) {
        ASSERT_EQ(input[i], output[i]);
    }
}

TEST(kernel_correctness, identity_filter_channels) {
    Pixel<3> input[16] = {
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12},
    {13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24},
    {25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36},
    {37, 38, 39}, {40, 41, 42}, {43, 44, 45}, {46, 47, 48}
    };

    Pixel<3> output[16] = {0};
    run_kernel<3>(IDENTITY_FILTER, 3, input, output, 4, 4);

    for(int i = 0; i < 16; i++) {
        ASSERT_EQ(input[i], output[i]);
    }
    CUDA_CHECK_ERROR("kernel execution");
}

TEST(kernel_correctness, simple_box_blur) {
    Pixel<3> input[9] = {
        {1, 1, 1}, {2, 2, 2}, {3, 3, 3},
        {4, 4, 4}, {5, 5, 5}, {6, 6, 6},
        {7, 7, 7}, {8, 8, 8}, {9, 9, 9}
    };
    Pixel<3> output[9] = {0};
    run_kernel<3>(BOX_BLUR_FILTER, 3, input, output, 3, 3);

    Pixel<3> expected[9] = {
        {12, 12, 12}, {21, 21, 21}, {16, 16, 16},
        {27, 27, 27}, {45, 45, 45}, {33, 33, 33},
        {24, 24, 24}, {39, 39, 39}, {28, 28, 28}
    };
    for(int i = 0; i < 9; i++) {
        ASSERT_EQ(expected[i], output[i]);
    }

    CUDA_CHECK_ERROR("kernel execution");
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

    // *** find the largest ***
    void *d_temp_storage_largest = nullptr;
    size_t temp_storage_bytes_largest = 0;

    cub::DeviceReduce::Reduce(
        d_temp_storage_largest, temp_storage_bytes_largest,
        d_input, d_largest, 9, max_op<3>(), Pixel<3>(INT_MIN));

    cudaMalloc(&d_temp_storage_largest, temp_storage_bytes_largest);

    cub::DeviceReduce::Reduce(
        d_temp_storage_largest, temp_storage_bytes_largest,
        d_input, d_largest, 9, max_op<3>(), Pixel<3>(INT_MIN));
    cudaDeviceSynchronize();

    cudaFree(d_temp_storage_largest);

    // *** find the smallest ***
    void *d_temp_storage_smallest = nullptr;
    size_t temp_storage_bytes_smallest = 0;

    CubDebugExit(cub::DeviceReduce::Reduce(
        d_temp_storage_smallest, temp_storage_bytes_smallest,
        d_input, d_smallest, 9, min_op<3>(), Pixel<3>{INT_MAX}));

    cudaMalloc(&d_temp_storage_smallest, temp_storage_bytes_smallest);

    CubDebugExit(cub::DeviceReduce::Reduce(
        d_temp_storage_smallest, temp_storage_bytes_smallest,
        d_input, d_smallest, 9, min_op<3>(), Pixel<3>{INT_MAX}));

    cudaDeviceSynchronize();
    cudaFree(d_temp_storage_smallest);

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}