#include "filter.h"

#include <stdio.h>

const int* identity_filter_data;
const int* edge_filter_data;
const int* sharpen_filter_data;
const int* box_blur_filter_data;
const int* gaussian_blur_filter_data;
const int* unsharp_masking_filter_data;
const int* high_pass_filter_data;
const int* emboss_filter_data;
const int* laplacian_filter_data;
const int* motion_blur_filter_data;

filter *identity_filter;
filter *edge_filter;
filter *sharpen_filter;
filter *box_blur_filter;
filter *gaussian_blur_filter;
filter *unsharp_masking_filter;
filter *high_pass_filter;
filter *emboss_filter;
filter *laplacian_filter;
filter *motion_blur_filter;

std::vector<const int*> basic_filter_data;
std::vector<filter*> basic_filters;

void force_initialize_filters() {
    identity_filter_data = new int[9]{
        0, 0, 0,
        0, 1, 0,
        0, 0, 0
    };
    edge_filter_data = new int[9]{
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
    };
    sharpen_filter_data = new int[9]{
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    box_blur_filter_data = new int[9]{
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    gaussian_blur_filter_data = new int[9]{
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
    };
    unsharp_masking_filter_data = new int[9]{
        -1, -1, -1,
        -1, 9, -1,
        -1, -1, -1
    };
    high_pass_filter_data = new int[9]{
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    emboss_filter_data = new int[9]{
        -2, -1, 0,
        -1, 1, 1,
        0, 1, 2
    };
    laplacian_filter_data = new int[9]{
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0
    };
    motion_blur_filter_data = new int[9]{
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    std::vector<const int*> filter_data = {
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

    filter *identity = new filter("Identity", identity_filter_data, 3);
    filter *edge = new filter("Edge", edge_filter_data, 3);
    filter *sharpen = new filter("Sharpen", sharpen_filter_data, 3);
    filter *box_blur = new filter("Box blur", box_blur_filter_data, 3);
    filter *gaussian_blur = new filter("Gaussian blur", gaussian_blur_filter_data, 3);
    filter *unsharp_masking = new filter("Unsharp masking", unsharp_masking_filter_data, 3);
    filter *high_pass = new filter("High pass", high_pass_filter_data, 3);
    filter *emboss = new filter("Emboss", emboss_filter_data, 3);
    filter *laplacian = new filter("Laplacian", laplacian_filter_data, 3);
    filter *motion_blur = new filter("Motion blur", motion_blur_filter_data, 3);

    std::vector<filter*> filters = {
        identity,
        edge,
        sharpen,
        box_blur,
        gaussian_blur,
        unsharp_masking,
        high_pass,
        emboss,
        laplacian,
        motion_blur
    };

    identity_filter = identity;
    edge_filter = edge;
    sharpen_filter = sharpen;
    box_blur_filter = box_blur;
    gaussian_blur_filter = gaussian_blur;
    unsharp_masking_filter = unsharp_masking;
    high_pass_filter = high_pass;
    emboss_filter = emboss;
    laplacian_filter = laplacian;
    motion_blur_filter = motion_blur;

    basic_filter_data = filter_data;
    basic_filters = filters;
}

const int* find_basic_filter_data(const char *name) {
    for(auto f : basic_filters) {
        if(strcmp(f->filter_name, name) == 0) {
            return f->filter_data;
        }
    }
    return nullptr;
}

const filter* find_basic_filter(const char *name) {
    for(auto f : basic_filters) {
        if(strcmp(f->filter_name, name) == 0) {
            return f;
        }
    }
    return nullptr;
}

// filter strength on an image is a function of the filters size relative to the images size
// expand_filter takes in percentage [0, 100], and expands the filter to the largest square that can fit
// if its 0 then its just default basic filter
// within the image, as a percentage of the image size
// i.e if percentage = 1, then the filter will be expanded to the largest square that can fit within 1% of the image
// returns true on success, false on failure
bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height, 
    const char *basic_filter_name, filter *destination) {
    if(percentage > 100) {
        return false;
        // since unsigned, percentage/width/height can't be negative
    }
    if(percentage == 0) {
        // set destination to the basic filter
        *destination = *find_basic_filter(basic_filter_name); 
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
    filter *expanded_filter = new filter();
    if(percentage > 100) {
        delete expanded_filter;
        return nullptr;
    }
    else if(percentage == 0) {
        // just return the basic filter
        *expanded_filter = *(find_basic_filter(basic_filter_name));
        return expanded_filter;
    }

    if(expand_filter(percentage, image_width, image_height, basic_filter_name, expanded_filter)) {
        return expanded_filter;
    }
    else {
        return nullptr;
    }

}