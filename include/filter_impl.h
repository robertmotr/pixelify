#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

struct filter_properties {
    bool                            expandable_size;
    bool                            adjustable_strength;
    unsigned char*                  sizes_avail;
    int                             num_sizes_avail;
    int                             lower_bound_strength;
    int                             upper_bound_strength;
    bool                            basic_filter;
};

class filter {
public:
    char*                           filter_name = nullptr;
    float*                          filter_data = nullptr;
    unsigned char*                  sizes = nullptr; // array of sizes for expandable filters. i.e identity can only be 3x3, but gaussian blur can be 3x3, 5x5, 7x7, etc
    unsigned char                   filter_dimension = 0;
    size_t                          name_size;
    struct filter_properties        properties;

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

        filter_dimension = other.filter_dimension;
        filter_data = new float[filter_dimension * filter_dimension];
        memcpy(filter_data, other.filter_data, filter_dimension * filter_dimension * sizeof(float));
        name_size = strlen(other.filter_name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, other.filter_name);

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
};

const float* find_basic_filter_data(const char *name);

const filter* find_basic_filter(const char *name);

const filter *create_filter(const char *basic_filter_name, unsigned char filter_dimension,
                      char filter_strength);

#endif // __FILTER__H__
