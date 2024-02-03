#include "filters.h"

void initialize_properties() {
    // Define filter kernels with corresponding properties
    identity_properties = new filter_properties {false, false, nullptr, 0, 0, 0};
    edge_properties = new filter_properties{false, true, nullptr, 0, 0, 100};
    sharpen_properties = new filter_properties {false, true, nullptr, 0, 0, 100};
    box_blur_properties = new filter_properties {true, true, new unsigned char[5]{3, 5, 7, 9, 11}, 5, -100, 100};
    gaussian_blur_properties = new filter_properties {true, true, new unsigned char[5]{3, 5, 7, 9, 11}, 0, 0, 100};
    unsharp_mask_properties = new filter_properties {true, true, nullptr, 0, 0, 100};
    high_pass_properties = new filter_properties {false, true, nullptr, 0, 0, 100};
    emboss_properties = new filter_properties {false, true, nullptr, 0, 0, 100};
    laplacian_properties = new filter_properties {false, true, nullptr, 0, 0, 100};
    motion_blur_properties = new filter_properties {true, true, new unsigned char[5]{3, 5, 7, 9, 11}, 5, 0, 100};
    horizontal_shear_properties = new filter_properties {false, true, nullptr, 0, -100, 100};
    vertical_shear_properties = new filter_properties {false, true, nullptr, 0, -100, 100};
    sobel_x_properties = new filter_properties {false, false, nullptr, 0, 0, 0};
    sobel_y_properties = new filter_properties {false, false, nullptr, 0, 0, 0};
    prewitt_x_properties = new filter_properties {false, false, nullptr, 0, 0, 0};
    prewitt_y_properties = new filter_properties {false, false, nullptr, 0, 0, 0};
}

void initialize_filter_data() {
    identity_filter_data = new float[9] {
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    edge_filter_data = new float[9] {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
    };
    sharpen_filter_data = new float[9] {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    box_blur_filter_data = new float[9] {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    gaussian_blur_filter_data = new float[9] {
        1/16, 2/16, 1/16,
        2/16, 4/16, 2/16,
        1/16, 2/16, 1/16
    };
    unsharp_mask_filter_data = new float[9] {
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1
    };
    high_pass_filter_data = new float[9] {
        -1, -1, -1,
        -1, 8, -1,
        -1, -1, -1
    };
    emboss_filter_data = new float[9] {
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2
    };
    laplacian_filter_data = new float[9] {
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0
    };
    motion_blur_filter_data = new float[9] {
        1/3, 0, 0,
        1/3, 0, 0,
        1/3, 0, 0
    };

    horizontal_shear_filter_data = new float[9] {
        1, 1, 0,
        0, 1, 0,
        0, 0, 1
    };
    vertical_shear_filter_data = new float[9] {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    sobel_x_filter_data = new float[9] {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    sobel_y_filter_data = new float[9] {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };
    prewitt_x_filter_data = new float[9] {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };
    prewitt_y_filter_data = new float[9] {
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1
    };
}

void initialize_filter_objects() {
    identity_filter = new filter("Identity", identity_filter_data, 3);
    edge_filter = new filter("Edge", edge_filter_data, 3);
    sharpen_filter = new filter("Sharpen", sharpen_filter_data, 3);
    box_blur_filter = new filter("Box Blur", box_blur_filter_data, 3);
    gaussian_blur_filter = new filter("Gaussian Blur", gaussian_blur_filter_data, 3);
    unsharp_mask_filter = new filter("Unsharp Mask", unsharp_mask_filter_data, 3);
    high_pass_filter = new filter("High Pass", high_pass_filter_data, 3);
    emboss_filter = new filter("Emboss", emboss_filter_data, 3);
    laplacian_filter = new filter("Laplacian", laplacian_filter_data, 3);
    motion_blur_filter = new filter("Motion Blur", motion_blur_filter_data, 3);
    horizontal_shear_filter = new filter("Horizontal Shear", horizontal_shear_filter_data, 3);
    vertical_shear_filter = new filter("Vertical Shear", vertical_shear_filter_data, 3);
    sobel_x_filter = new filter("Sobel X", sobel_x_filter_data, 3);
    sobel_y_filter = new filter("Sobel Y", sobel_y_filter_data, 3);
    prewitt_x_filter = new filter("Prewitt X", prewitt_x_filter_data, 3);
    prewitt_y_filter = new filter("Prewitt Y", prewitt_y_filter_data, 3);
}

__attribute__((constructor))
const filter** init_filters() {
    // init basic filter data
    initialize_filter_data();

    // init filter properties
    initialize_properties();

    // create filters
    initialize_filter_objects();

    basic_filter_data_array = new const float* [BASIC_FILTER_SIZE] {
        identity_filter_data,
        edge_filter_data,
        sharpen_filter_data,
        box_blur_filter_data,
        gaussian_blur_filter_data,
        unsharp_mask_filter_data,
        high_pass_filter_data,
        emboss_filter_data,
        laplacian_filter_data,
        motion_blur_filter_data,
        horizontal_shear_filter_data,
        vertical_shear_filter_data,
        sobel_x_filter_data,
        sobel_y_filter_data,
        prewitt_x_filter_data,
        prewitt_y_filter_data
    };

    basic_filters_array = new const filter* [BASIC_FILTER_SIZE] {
        identity_filter,
        edge_filter,
        sharpen_filter,
        box_blur_filter,
        gaussian_blur_filter,
        unsharp_mask_filter,
        high_pass_filter,
        emboss_filter,
        laplacian_filter,
        motion_blur_filter,
        horizontal_shear_filter,
        vertical_shear_filter,
        sobel_x_filter,
        sobel_y_filter,
        prewitt_x_filter,
        prewitt_y_filter
    };

    return basic_filters_array;
}