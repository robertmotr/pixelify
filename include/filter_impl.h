#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

#define OUT_OF_BOUNDS           -1

struct filter_args {
    bool                        normalize; // false means we clamp values to [0, 255] to be able to display them,
                                           // true means perform linear normalization instead
    bool                        invert; // invert the image i.e 255 - pixel_value
    bool                        threshold; // threshold the image i.e if pixel_value > 127 then pixel_value = 255 else pixel_value = 0
    bool                        conversion; // convert the image to grayscale using the formula: 0.299 * red + 0.587 * green + 0.114 * blue
    // values below are expected to be in [-100, 100] range
    // 0 means do nothing, 0 < x < 100 means increase values by x%, 0 > x > -100 means decrease values by x%
    unsigned char               filter_strength; // how much of the filter to apply [0, 100]
    unsigned char               dimension; // filter dimension
    char                        red_shift;       
    char                        green_shift; 
    char                        blue_shift; 
    char                        alpha_shift; 
    char                        brightness;
    // chosen by colour picker
    unsigned char tint[4] =     {0, 0, 0, 0}; // [red, green, blue, alpha]
    float                       blend_factor; // how much of the tint to apply [0, 1]
    unsigned char               passes; // how many times to apply the filter
};

struct filter_properties {
    bool                            expandable_size;
    bool                            adjustable_strength;
    unsigned char*                  sizes_avail; // array of sizes for expandable filters. i.e identity can only be 3x3, but gaussian blur can be 3x3, 5x5, 7x7, etc
    int                             num_sizes_avail;
    int                             lower_bound_strength;
    int                             upper_bound_strength;
    bool                            basic_filter;
};

class filter {
public:
    char*                           filter_name = nullptr;
    float*                          filter_data = nullptr;
    unsigned char                   filter_dimension = 0;
    size_t                          name_size;
    struct filter_properties*       properties = nullptr;

    // default
    filter() : filter_dimension(0), filter_data(nullptr) {
        size_t size = strlen("NULL") + 1;
        filter_name = new char[size];
        strcpy(filter_name, "NULL");
        this->name_size = size;
    }

    // constructor with name/data/dimension
    filter(const char* name, float *data, unsigned int dimension) {
        filter_dimension = dimension;
        filter_data = data;

        name_size = strlen(name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, name);
    }

    // constructor with name/data/dimension but const
    filter(const char* name, const float *data, unsigned int dimension) {
        filter_dimension = dimension;
        filter_data = const_cast<float*>(data);

        name_size = strlen(name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, name);
    }

    // constructor with just name
    filter(const char* name)
        : filter_dimension(0), filter_data(nullptr) {
        size_t size = strlen(name) + 1;

        filter_name = new char[size];
        strcpy(filter_name, name);
        this->name_size = size;
    }

    ~filter() {
        if (filter_data != nullptr) {
            delete[] filter_data;
            filter_data = nullptr;
        }
        if (filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }
        if(properties != nullptr) {
            delete properties;
            properties = nullptr;
        }
    }

    bool operator==(const filter &other) const {
        if (filter_name != other.filter_name) {
            return false;
        }
        if (filter_dimension != other.filter_dimension) {
            return false;
        }
        for (unsigned int i = 0; i < filter_dimension * filter_dimension; i++) {
            if (filter_data[i] != other.filter_data[i]) {
                return false;
            }
        }
        if (properties != other.properties) {
            return false;
        }

        return true;
    }

    filter& operator=(const filter &other) {
        if (this == &other) {
            return *this;
        }

        if(filter_data != nullptr) {
            delete[] filter_data;
            filter_data = nullptr;
        }
        if(filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }
        if(properties != nullptr) {
            delete properties;
            properties = nullptr;
        }

        filter_dimension = other.filter_dimension;
        filter_data = new float[filter_dimension * filter_dimension];
        memcpy(filter_data, other.filter_data, filter_dimension * filter_dimension * sizeof(float));
        name_size = strlen(other.filter_name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, other.filter_name);
        properties = new filter_properties;
        if(other.properties != nullptr) {
            memcpy(properties, other.properties, sizeof(filter_properties));
        }

        return *this;
    }

    filter(const filter& other) {
        filter_data = new float[other.filter_dimension * other.filter_dimension];
        memcpy(filter_data, other.filter_data, other.filter_dimension * other.filter_dimension * sizeof(float));
        filter_dimension = other.filter_dimension;
        name_size = strlen(other.filter_name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, other.filter_name);
    }

    void set_properties(filter_properties* prop) {
        properties = prop;
    }

    void set_properties(const filter_properties* prop) {
        properties = const_cast<filter_properties*>(prop);
    }
};

const float* find_basic_filter_data(const char *name);

const filter* find_basic_filter(const char *name);

const filter *create_filter(const char *basic_filter_name, unsigned char filter_dimension,
                      char filter_strength);

#endif // __FILTER__H__
