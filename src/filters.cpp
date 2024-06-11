#include "filters.h"
#include "filter_impl.h"
#include "kernel_formulas.h"

// arrays that hold actual filter values
const float* identity_filter_data;
const float* edge_filter_data;
const float* sharpen_filter_data;
const float* box_blur_filter_data;
const float* gaussian_blur_filter_data;
const float* unsharp_mask_filter_data;
const float* emboss_filter_data;
const float* laplacian_filter_data;
const float* motion_blur_filter_data;
const float* horizontal_shear_filter_data;
const float* vertical_shear_filter_data;
const float* sobel_x_filter_data;
const float* sobel_y_filter_data;
const float* prewitt_x_filter_data;
const float* prewitt_y_filter_data;

// filter objects
filter *identity_filter;
filter *edge_filter;
filter *sharpen_filter;
filter *box_blur_filter;
filter *gaussian_blur_filter;
filter *unsharp_mask_filter;
filter *emboss_filter;
filter *laplacian_filter;
filter *motion_blur_filter;
filter *horizontal_shear_filter;
filter *vertical_shear_filter;
filter *sobel_x_filter;
filter *sobel_y_filter;
filter *prewitt_x_filter;
filter *prewitt_y_filter;

// filter properties
struct filter_properties* identity_properties;
struct filter_properties* edge_properties;
struct filter_properties* sharpen_properties;
struct filter_properties* box_blur_properties;
struct filter_properties* gaussian_blur_properties;
struct filter_properties* unsharp_mask_properties;
struct filter_properties* emboss_properties;
struct filter_properties* laplacian_properties;
struct filter_properties* motion_blur_properties;
struct filter_properties* horizontal_shear_properties;
struct filter_properties* vertical_shear_properties;
struct filter_properties* sobel_x_properties;
struct filter_properties* sobel_y_properties;
struct filter_properties* prewitt_x_properties;
struct filter_properties* prewitt_y_properties;

const float** basic_filter_data_array;
const filter** basic_filters_array;

void initialize_properties() {
    // Define filter kernels with corresponding properties
    identity_properties =           new filter_properties {false, false, nullptr, 0, 0, 0, true};
    edge_properties =               new filter_properties {false, false, nullptr, 0, 0, 0, true};
    sharpen_properties =            new filter_properties {false, true, nullptr, 0, 0, 100, true};
    box_blur_properties =           new filter_properties {true, true, new unsigned char[7]{3, 5, 7, 9, 11, 13, 15}, 7, -100, 100, true};
    gaussian_blur_properties =      new filter_properties {true, true, new unsigned char[7]{3, 5, 7, 9, 11, 13, 15}, 7, 0, 100, true};
    unsharp_mask_properties =       new filter_properties {true, true, new unsigned char[7]{3, 5, 7, 9, 11, 13, 15}, 7, 0, 10, true};
    emboss_properties =             new filter_properties {false, true, nullptr, 0, 0, 100, true};
    laplacian_properties =          new filter_properties {false, true, nullptr, 0, 0, 100, true};
    motion_blur_properties =        new filter_properties {true, true, new unsigned char[5]{3, 5, 7, 9, 11}, 5, 0, 100, true};
    horizontal_shear_properties =   new filter_properties {false, true, nullptr, 0, -100, 100, true};
    vertical_shear_properties =     new filter_properties {false, true, nullptr, 0, -100, 100, true};
    sobel_x_properties =            new filter_properties {false, false, nullptr, 0, 0, 0, true};
    sobel_y_properties =            new filter_properties {false, false, nullptr, 0, 0, 0, true};
    prewitt_x_properties =          new filter_properties {false, false, nullptr, 0, 0, 0, true};
    prewitt_y_properties =          new filter_properties {false, false, nullptr, 0, 0, 0, true};
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
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    unsharp_mask_filter_data = new float[9] {
        -1, -1, -1,
        -1, 9, -1,
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
        1, 0.01, 0,
        0, 1, 0,
        0, 0, 1
    };
    vertical_shear_filter_data = new float[9] {
        1, 0, 0,
        0.01, 1, 0,
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
    identity_filter->set_properties(identity_properties);
    edge_filter = new filter("Edge", edge_filter_data, 3);
    edge_filter->set_properties(edge_properties);
    sharpen_filter = new filter("Sharpen", sharpen_filter_data, 3);
    sharpen_filter->set_properties(sharpen_properties);
    box_blur_filter = new filter("Box Blur", box_blur_filter_data, 3);
    box_blur_filter->set_properties(box_blur_properties);
    gaussian_blur_filter = new filter("Gaussian Blur", gaussian_blur_filter_data, 3);
    gaussian_blur_filter->set_properties(gaussian_blur_properties);
    unsharp_mask_filter = new filter("Unsharp Mask", unsharp_mask_filter_data, 3);
    unsharp_mask_filter->set_properties(unsharp_mask_properties);
    emboss_filter = new filter("Emboss", emboss_filter_data, 3);
    emboss_filter->set_properties(emboss_properties);
    laplacian_filter = new filter("Laplacian", laplacian_filter_data, 3);
    laplacian_filter->set_properties(laplacian_properties);
    motion_blur_filter = new filter("Motion Blur", motion_blur_filter_data, 3);
    motion_blur_filter->set_properties(motion_blur_properties);
    horizontal_shear_filter = new filter("Horizontal Shear", horizontal_shear_filter_data, 3);
    horizontal_shear_filter->set_properties(horizontal_shear_properties);
    vertical_shear_filter = new filter("Vertical Shear", vertical_shear_filter_data, 3);
    vertical_shear_filter->set_properties(vertical_shear_properties);
    sobel_x_filter = new filter("Sobel X", sobel_x_filter_data, 3);
    sobel_x_filter->set_properties(sobel_x_properties);
    sobel_y_filter = new filter("Sobel Y", sobel_y_filter_data, 3);
    sobel_y_filter->set_properties(sobel_y_properties);
    prewitt_x_filter = new filter("Prewitt X", prewitt_x_filter_data, 3);
    prewitt_x_filter->set_properties(prewitt_x_properties);
    prewitt_y_filter = new filter("Prewitt Y", prewitt_y_filter_data, 3);
    prewitt_y_filter->set_properties(prewitt_y_properties);
}

const filter** init_filters() {
    // init basic filter data
    initialize_filter_data();

    // init filter properties
    initialize_properties();

    // create filters
    initialize_filter_objects();

    init_kernel_formulas();

    basic_filter_data_array = new const float* [BASIC_FILTER_SIZE] {
        identity_filter_data,
        edge_filter_data,
        sharpen_filter_data,
        box_blur_filter_data,
        gaussian_blur_filter_data,
        unsharp_mask_filter_data,
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