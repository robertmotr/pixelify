#ifndef FILTERS_H
#define FILTERS_H

#include "filter_impl.h"
#include "kernel_formulas.h"

#define BASIC_FILTER_SIZE 16

// arrays that hold actual filter values
extern const float* identity_filter_data;
extern const float* edge_filter_data;
extern const float* sharpen_filter_data;
extern const float* box_blur_filter_data;
extern const float* gaussian_blur_filter_data;
extern const float* unsharp_mask_filter_data;
extern const float* high_pass_filter_data;
extern const float* emboss_filter_data;
extern const float* laplacian_filter_data;
extern const float* motion_blur_filter_data;
extern const float* horizontal_shear_filter_data;
extern const float* vertical_shear_filter_data;
extern const float* sobel_x_filter_data;
extern const float* sobel_y_filter_data;
extern const float* prewitt_x_filter_data;
extern const float* prewitt_y_filter_data;

// filter objects
extern const filter *identity_filter;
extern const filter *edge_filter;
extern const filter *sharpen_filter;
extern const filter *box_blur_filter;
extern const filter *gaussian_blur_filter;
extern const filter *unsharp_mask_filter;
extern const filter *high_pass_filter;
extern const filter *emboss_filter;
extern const filter *laplacian_filter;
extern const filter *motion_blur_filter;
extern const filter *horizontal_shear_filter;
extern const filter *vertical_shear_filter;
extern const filter *sobel_x_filter;
extern const filter *sobel_y_filter;
extern const filter *prewitt_x_filter;
extern const filter *prewitt_y_filter;

// filter properties
extern struct filter_properties* identity_properties;
extern struct filter_properties* edge_properties;
extern struct filter_properties* sharpen_properties;
extern struct filter_properties* box_blur_properties;
extern struct filter_properties* gaussian_blur_properties;
extern struct filter_properties* unsharp_mask_properties;
extern struct filter_properties* high_pass_properties;
extern struct filter_properties* emboss_properties;
extern struct filter_properties* laplacian_properties;
extern struct filter_properties* motion_blur_properties;
extern struct filter_properties* horizontal_shear_properties;
extern struct filter_properties* vertical_shear_properties;
extern struct filter_properties* sobel_x_properties;
extern struct filter_properties* sobel_y_properties;
extern struct filter_properties* prewitt_x_properties;
extern struct filter_properties* prewitt_y_properties;

extern const float** basic_filter_data_array;
extern const filter** basic_filters_array;

// returns ptr to array of filter (object) ptrs
const filter** init_filters();

#endif