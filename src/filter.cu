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
const int* sobel_x_filter_data;
const int* sobel_y_filter_data;

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
    filter *unsharp_masking = new filter("Unsharp mask", unsharp_masking_filter_data, 3);
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

filter *create_filter_from_strength(const char *basic_filter_name, unsigned int image_width,
    unsigned int image_height, unsigned char strength) {
    filter *expanded_filter = new filter();
    if(strength > 100) {
        delete expanded_filter;
        return nullptr;
    }
    else if(strength == 0) {
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
    else {
        // valid strength size

    }
}

struct filter_properties get_filter_properties(const char *filter_name) {
    struct filter_properties properties;
    memset(&properties, 0, sizeof(struct filter_properties));
    
    if(strcmp(filter_name, "NULL") == 0) {
        printf("Error: filter not found, is NULL\n");
    }
    else if(strcmp(filter_name, "Identity") == 0) {
        // already memset to be 0 
    }
    else if(strcmp(filter_name, "Edge") == 0) {
        // already memset to be 0
    }
    else if(strcmp(filter_name, "Sharpen") == 0) {
        properties.adjustable_strength = true;
        properties.expandable_size = false;
        properties.lower_bound_strength = 0;
        properties.upper_bound_strength = 100;
    }
    else if(strcmp(filter_name, "Box blur") == 0) {
        properties.adjustable_strength = false;
        properties.expandable_size = true;
        properties.num_sizes = 10;
        properties.sizes = new unsigned char[properties.num_sizes]{
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        };
    }
    else if(strcmp(filter_name, "Gaussian blur") == 0) {
        properties.adjustable_strength = true;
        properties.expandable_size = true;
        properties.num_sizes = 5;
        properties.sizes = new unsigned char[properties.num_sizes]{
            3, 5, 7, 9, 11
        };
    }
    else if(strcmp(filter_name, "Unsharp mask") == 0) {
        
    }
    else if(strcmp(filter_name, "High pass") == 0) {

    }
    else if(strcmp(filter_name, "Low pass") == 0) {

    }
    else if(strcmp(filter_name, "Emboss") == 0) {

    }
    else if(strcmp(filter_name, "Laplacian") == 0) {

    }
    else if(strcmp(filter_name, "Motion blur") == 0) {
        
    }
    else if(strcmp(filter_name, "Horizontal shear") == 0) {

    }
    else if(strcmp(filter_name, "Vertical shear") == 0) {

    }
    else if(strcmp(filter_name, "Sobel") == 0) {

    }
    else if(strcmp(filter_name, "Prewitt") == 0) {

    }
    else {
        printf("Error: filter not found\n");
    }
    return properties;
}