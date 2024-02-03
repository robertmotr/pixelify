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
    if(strcmp(filter_name, "NULL") == 0) {
        return nullptr;
    }
    else if(find_basic_filter(filter_name) == nullptr) {
        return nullptr;
    }
    else if(strcmp(filter_name, "Identity") == 0) {
        return find_basic_filter("Identity");
    }
    else {
        // if we can match a basic filter to strength + dimension
        // return it, otherwise we create our own
        const filter *basic_filter = find_basic_filter(filter_name);
        if(basic_filter != nullptr) {
            if(basic_filter->filter_dimension == filter_dimension &&
               (!basic_filter->properties->adjustable_strength && filter_strength != 0)) {
                return basic_filter;
            }
        }

        kernel_formula_fn fn_ptr = kernel_formulas->at(filter_name);
        float *filter_data = new float[filter_dimension * filter_dimension];
        for (int i = 0; i < filter_dimension; i++) {
            for(int j = 0; j < filter_dimension; j++) {
                filter_data[i * filter_dimension + j] = fn_ptr(i, j, filter_strength, filter_dimension);
            }
        }

        filter *f = new filter(filter_name, filter_data, filter_dimension);
        filter_properties *props = new filter_properties();
        props->basic_filter = false;
        f->properties = props;
        return f;
    }
}



