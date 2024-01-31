#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

#define FILTER_DIMENSION 3

class filter {
public:

    char*                           filter_name;
    int*                            filter_data;
    unsigned int                    filter_dimension = 0;
    size_t                          name_size;

    // default
    filter() : filter_dimension(0), filter_data(nullptr) {
        size_t size = strlen("NULL") + 1;
        filter_name = new char[size];
        strcpy(filter_name, "NULL");
        this->name_size = size;
    }

    // constructor with name/data/dimension
    filter(const char* name, int *data, unsigned int dimension) {
        filter_dimension = dimension;
        filter_data = data;

        name_size = strlen(name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, name);
    }

    // constructor with name/data/dimension but const
    filter(const char* name, const int *data, unsigned int dimension) {
        filter_dimension = dimension;
        filter_data = const_cast<int*>(data);

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
        filter_data = new int[filter_dimension * filter_dimension];
        memcpy(filter_data, other.filter_data, filter_dimension * filter_dimension * sizeof(int));
        name_size = strlen(other.filter_name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, other.filter_name);

        return *this;
    }

    filter(const filter& other) {
        filter_data = new int[other.filter_dimension * other.filter_dimension];
        memcpy(filter_data, other.filter_data, other.filter_dimension * other.filter_dimension * sizeof(int));
        filter_dimension = other.filter_dimension;
        name_size = strlen(other.filter_name) + 1;
        filter_name = new char[name_size];
        strcpy(filter_name, other.filter_name);
    }
};

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

extern const filter *identity_filter;
extern const filter *edge_filter;
extern const filter *sharpen_filter;
extern const filter *box_blur_filter;
extern const filter *gaussian_blur_filter;
extern const filter *unsharp_mask_filter;
extern const filter *high_pass_filter;
extern const filter *emboss_filter;
extern const filter *sobel_filter;
extern const filter *laplacian_filter;
extern const filter *motion_blur_filter;

extern const int** basic_filter_data_array;
extern const filter** basic_filters_array;
extern const int filter_array_size;

// allegedly this attribute forces the function to be called before main
void force_initialize_filters() __attribute__((constructor));

const int* find_basic_filter_data(const char *name);
const filter* find_basic_filter(const char *name);
const std::vector<filter> get_filters();

filter *create_filter_from_strength(const char *basic_filter_name, unsigned int image_width,
 unsigned int image_height, unsigned char percentage);

// filter strength on an image is a function of the filters size relative to the images size
// expand_filter takes in percentage [0, 100], and expands the filter to the largest rectangle that can fit
// within the image, as a percentage of the image size
// i.e if percentage = 1, then the filter will be expanded to the largest rectangle that can fit within 1% of the image
// returns true on success, false on failure
bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height, 
    const char *basic_filter_name, filter *destination);

filter** get_all_filters();

int** get_all_filter_data();
#endif // __FILTER__H__
