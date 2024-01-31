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

const filter *identity_filter;
const filter *edge_filter;
const filter *sharpen_filter;
const filter *box_blur_filter;
const filter *gaussian_blur_filter;
const filter *unsharp_masking_filter;
const filter *high_pass_filter;
const filter *emboss_filter;
const filter *laplacian_filter;
const filter *motion_blur_filter;

const int** basic_filter_data_array;
const filter** basic_filters_array;
const int filter_array_size = 10;

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

    basic_filter_data_array = new const int*[10] {
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

    basic_filters_array = new const filter*[10] {
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

}

const int* find_basic_filter_data(const char *name) {
    for(int i = 0; i < filter_array_size; i++) {
        const filter *f = basic_filters_array[i];
        if(strcmp(f->filter_name, name) == 0) {
            return basic_filter_data_array[i];
        }
    }
    return nullptr;
}

const filter* find_basic_filter(const char *name) {
    for(int i = 0; i < filter_array_size; i++) {
        const filter *f = basic_filters_array[i];
        if(strcmp(f->filter_name, name) == 0) {
            return f;
        }
    }
    return nullptr;
}

int find_largest_square(int m, int n, unsigned char percentage) {
    int area = m * n;
    int square_dimension = static_cast<int>(sqrt(percentage / 100.0 * area / 9));
    square_dimension -= square_dimension % 3;
    return square_dimension;
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

    int best_dimension = find_largest_square(image_width, image_height, percentage);

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
        const filter *basic_filter = find_basic_filter(basic_filter_name);
        if(basic_filter == nullptr) {
            delete expanded_filter;
            printf("Error: basic filter not found\n");
            return nullptr;
        }
        else {
            *expanded_filter = *basic_filter;
            return expanded_filter;
        }
    }

    if(expand_filter(percentage, image_width, image_height, basic_filter_name, expanded_filter)) {
        return expanded_filter;
    }
    else {
        return nullptr;
    }

}