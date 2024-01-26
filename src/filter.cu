#include "filter.h"

const int* identity_filter_data = new int[9]{
    0, 0, 0,
    0, 1, 0,
    0, 0, 0
};
const int* edge_filter_data = new int[9]{
    0, 1, 0,
    1, -4, 1,
    0, 1, 0
};
const int* sharpen_filter_data = new int[9]{
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};
const int* box_blur_filter_data = new int[9]{
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
};
const int* gaussian_blur_filter_data = new int[9]{
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};
const int* unsharp_masking_filter_data = new int[9]{
    -1, -1, -1,
    -1, 9, -1,
    -1, -1, -1
};
const int* high_pass_filter_data = new int[9]{
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
};
const int* emboss_filter_data = new int[9]{
    -2, -1, 0,
    -1, 1, 1,
    0, 1, 2
};
const int* laplacian_filter_data = new int[9]{
    0, -1, 0,
    -1, 4, -1,
    0, -1, 0
};
const int* motion_blur_filter_data = new int[9]{
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
};

const std::vector<const int*> basic_filter_data = {
    identity_filter_data,
    edge_filter_data,
    sharpen_filter_data,
    box_blur_filter_data,
    gaussian_blur_filter_data,
    unsharp_masking_filter_data,
    high_pass_filter_data,
    emboss_filter_data,
    laplacian_filter_data,
    motion_blur_filter_data
};

const filter identity_filter = filter("Identity Filter", identity_filter_data, 3);
const filter edge_filter = filter("Edge Filter", edge_filter_data, 3);
const filter sharpen_filter = filter("Sharpen Filter", sharpen_filter_data, 3);
const filter box_blur_filter = filter("Box Blur Filter", box_blur_filter_data, 3);
const filter gaussian_blur_filter = filter("Gaussian Blur Filter", gaussian_blur_filter_data, 3);
const filter unsharp_masking_filter = filter("Unsharp Masking Filter", unsharp_masking_filter_data, 3);
const filter high_pass_filter = filter("High Pass Filter", high_pass_filter_data, 3);
const filter emboss_filter = filter("Emboss Filter", emboss_filter_data, 3);
const filter laplacian_filter = filter("Laplacian Filter", laplacian_filter_data, 3);
const filter motion_blur_filter = filter("Motion Blur Filter", motion_blur_filter_data, 3);

const std::vector<filter> basic_filters = {
    identity_filter,
    edge_filter,
    sharpen_filter,
    box_blur_filter,
    gaussian_blur_filter,
    unsharp_masking_filter,
    high_pass_filter,
    emboss_filter,
    laplacian_filter,
    motion_blur_filter
};

const int* find_basic_filter_data(const char *name) {
    for(auto f : basic_filters) {
        if(strcmp(f.filter_name, name) == 0) {
            return f.filter_data;
        }
    }
    return nullptr;
}

const filter find_basic_filter(const char *name) {
    for(auto f : basic_filters) {
        if(strcmp(f.filter_name, name) == 0) {
            return f;
        }
    }
    return filter(); // null filter
}

// filter strength on an image is a function of the filters size relative to the images size
// expand_filter takes in percentage [0, 100], and expands the filter to the largest rectangle that can fit
// within the image, as a percentage of the image size
// i.e if percentage = 1, then the filter will be expanded to the largest rectangle that can fit within 1% of the image
// returns true on success, false on failure
bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height, 
    const char *basic_filter_name, filter *destination) {
    if(percentage > 100) {
        return false;
        // since unsigned, percentage/width/height can't be negative
    }
    if(percentage == 0) {
        return true;
    }

    double desired_area = (percentage / 100.0) * image_width * image_height;

    // Initialize variables to store the best result
    double best_diff = std::numeric_limits<double>::infinity();
    unsigned int best_dimension = 0;

    // Iterate over all possible rectangle sizes
    for (unsigned int dim = 3; dim < std::min(image_width, image_height); dim += 3) {
    
        double current_area = dim * dim;

        // difference between the current area and the desired area
        double diff = std::abs(current_area - desired_area);

        // update best result if the current is closer to desired
        if (diff < best_diff) {
            best_diff = diff;
            best_dimension = dim;
        }
    }

    // it could be possible best_dimension > image_width/image_height
    // in that case scale it back a bit such that it still is within the image and
    // is divisible by 3x3 squares
    while((best_dimension > image_width ||
            best_dimension > image_height ||
            best_dimension % 3 != 0) && 
            best_dimension > 3) {
        best_dimension -= 3;
    }
    assert(best_dimension % 3 == 0 && best_dimension > 0);
    assert(best_dimension <= image_width && best_dimension <= image_height);

    const int *copy_data = find_basic_filter_data(basic_filter_name);

    int *new_filter_data = new int[best_dimension * best_dimension];
    for(int i = 0; i < best_dimension * best_dimension; i += 9) {
        memcpy(new_filter_data + i, copy_data, FILTER_DIMENSION * FILTER_DIMENSION * sizeof(int));
    }

    *destination = filter(basic_filter_name, new_filter_data, best_dimension);

    return true;
}

filter *create_filter_from_strength(const char *basic_filter_name, unsigned int image_width,
    unsigned int image_height, unsigned char percentage) {
    
    if(percentage > 100 || percentage == 0) {
        return nullptr;
    }

    filter *expanded_filter = new filter("Expanded filter");

    if(expand_filter(percentage, image_width, image_height, basic_filter_name, expanded_filter)) {
        return expanded_filter;
    }
    else {
        return nullptr;
    }

}

const std::vector<filter> get_filters() {
    return basic_filters;
} 