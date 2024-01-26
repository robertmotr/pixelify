#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

#define FILTER_DIMENSION 3

extern const int* identity_filter_data;
extern const int* edge_filter_data;
extern const int* sharpen_filter_data;
extern const int* box_blur_filter_data;
extern const int* gaussian_blur_filter_data;
extern const int* unsharp_mask_filter_data;
extern const int* high_pass_filter_data;
extern const int* emboss_filter_data;
extern const int* sobel_filter_data;
extern const int* laplacian_filter_data;
extern const int* motion_blur_filter_data;

extern const filter identity_filter;
extern const filter edge_filter;
extern const filter sharpen_filter;
extern const filter box_blur_filter;
extern const filter gaussian_blur_filter;
extern const filter unsharp_mask_filter;
extern const filter high_pass_filter;
extern const filter emboss_filter;
extern const filter sobel_filter;
extern const filter laplacian_filter;
extern const filter motion_blur_filter;

extern const std::vector<const int*> basic_filter_data;

class filter {
public:

    char*                           filter_name;
    int*                            filter_data;
    unsigned int                    filter_dimension;
    size_t                          name_size;

    // default
    filter() : filter_dimension(0), filter_data(nullptr) {
        size_t size = strlen("Null filter") + 1;
        filter_name = new char[size];
        strcpy(filter_name, "Null filter");
        this->name_size = size;
    }

    // constructor with name/data/dimension
    filter(const char* name, int *data, unsigned int dimension)
        : filter_dimension(dimension), filter_data(data) {
        size_t size = strlen(name) + 1;

        if(filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }

        filter_name = new char[size];
        strcpy(filter_name, name);
        this->name_size = size;
    }

    // constructor with just name
    filter(const char* name)
        : filter_dimension(0), filter_data(nullptr) {
        size_t size = strlen(name) + 1;

        filter_name = new char[size];
        strcpy(filter_name, name);
        this->name_size = size;
    }

    // basic filter initializer
    filter(const char* name, const int* data, unsigned int dimension)
        : filter_dimension(dimension) {
        size_t size = strlen(name) + 1;

        if(filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }

        filter_name = new char[size];
        strcpy(filter_name, name);
        this->name_size = size;
    }

    // constructor with name/data specified
    filter(const char* name, unsigned int dimension)
        : filter_dimension(dimension), filter_data(new int[dimension * dimension]) {
        size_t size = strlen(name) + 1;

        if(filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }

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
        if (filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }
        if (filter_data != other.filter_data) {
            if(filter_data != nullptr) {
                delete[] filter_data;
                filter_data = nullptr;
            }
            filter_data = new int[other.filter_dimension * other.filter_dimension]; 
            memcpy(filter_data, other.filter_data, other.filter_dimension * other.filter_dimension * sizeof(int));
        }
        
        return *this;
    }
};

const int* find_basic_filter_data(const char *name);
const filter find_basic_filter(const char *name);

filter *create_filter_from_strength(const char *name, unsigned int image_width,
 unsigned int image_height, unsigned char percentage);

// filter strength on an image is a function of the filters size relative to the images size
// expand_filter takes in percentage [0, 100], and expands the filter to the largest rectangle that can fit
// within the image, as a percentage of the image size
// i.e if percentage = 1, then the filter will be expanded to the largest rectangle that can fit within 1% of the image
// returns true on success, false on failure
bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height, 
    const char *basic_filter_name, filter *destination);
#endif // __FILTER__H__
