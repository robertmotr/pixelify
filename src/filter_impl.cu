#include "filter_impl.h"
#include "filters.h"

#include <stdio.h>

int kronecker_delta(int i, int j) {
    return i == j ? 1 : 0;
}

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
    if(strcmp(filter_name, "NULL") == 0) {
        return nullptr;
    }
    else if(strcmp(filter_name, "Identity") == 0) {
        const filter *identity = find_basic_filter("Identity");
        filter *f = new filter("Identity");
        f->filter_data = new float[filter_dimension * filter_dimension];
        for(int i = 0; i < filter_dimension; i++) {
            for(int j = 0; j < filter_dimension; j++) {
                f->filter_data[i * filter_dimension + j] = identity->filter_data[i * filter_dimension + j];
            }
        }
        f->filter_dimension = identity->filter_dimension;
        return f; 
    }
    else {
        // function pointer = formula_dictionary[filter_name]
        // float *filter_data = new float[filter_dimension * filter_dimension];
        // function_pointer(filter_data, filter_dimension, filter_strength);
    }
}



