#include "filter_impl.h"
#include "filters.h"
#include "kernel_formulas.h"

#include <stdio.h>

const float* find_basic_filter_data(const char *name) {
    for(int i = 0; i < BASIC_FILTER_SIZE; i++) {
        const filter *f = basic_filters_array[i];
        if(strcmp(f->filter_name, name) == 0) {
            return basic_filter_data_array[i];
        }
    }
    return nullptr;
}

const filter* find_basic_filter(const char *name) {
    for(int i = 0; i < BASIC_FILTER_SIZE; i++) {
        const filter *f = basic_filters_array[i];
        if(strcmp(f->filter_name, name) == 0) {
            return f;
        }
    }
    return nullptr;
}

const filter* create_filter(const char *filter_name, unsigned char filter_dimension,
                      char filter_strength) {
    const filter *basic_filter = find_basic_filter(filter_name);
    if(basic_filter == nullptr) {
        return nullptr;
    }
    // if we can match a basic filter to strength + dimension
    // return it, otherwise we create our own

    else if(basic_filter->filter_dimension == filter_dimension &&
            filter_strength == 0) {
        #ifdef _DEBUG
            // print filter data of new_filter against its size nicely formatted
            printf("Filter data for %s with dimension %d and strength %d\n", filter_name, filter_dimension, filter_strength);
            // print as a matrix
            for (int i = 0; i < filter_dimension; i++) {
                for(int j = 0; j < filter_dimension; j++) {
                    printf("%f ", basic_filter->filter_data[i * filter_dimension + j]);
                }
                printf("\n");
            }
            printf("\n");  
        #endif
        
        return basic_filter;
    }
    else {
        kernel_formula_fn fn_ptr = kernel_formulas->at(filter_name);
        float *filter_data = new float[filter_dimension * filter_dimension];
        for (int i = 0; i < filter_dimension; i++) {
            for(int j = 0; j < filter_dimension; j++) {
                filter_data[i * filter_dimension + j] = fn_ptr(i, j, filter_strength, filter_dimension);
            }
        }

        filter* new_filter = new filter(filter_name, filter_data, filter_dimension);
        new_filter->properties = new filter_properties;
        new_filter->properties->basic_filter = false;

        #ifdef _DEBUG
            // print filter data of new_filter against its size nicely formatted
            printf("Filter data for %s with dimension %d and strength %d\n", filter_name, filter_dimension, filter_strength);
            // print as a matrix
            for (int i = 0; i < filter_dimension; i++) {
                for(int j = 0; j < filter_dimension; j++) {
                    printf("%f ", filter_data[i * filter_dimension + j]);
                }
                printf("\n");
            }
            printf("\n");   
        #endif     
        
        return new_filter;
    }
}



