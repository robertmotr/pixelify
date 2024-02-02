#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

#define FILTER_DIMENSION 3

struct filter_properties {
    bool expandable_size;
    bool adjustable_strength;
    unsigned char* sizes;
    int num_sizes;
    int lower_bound_strength;
    int upper_bound_strength;
}

class filter {
public:

    char*                           filter_name = nullptr;
    int*                            filter_data = nullptr;
    unsigned char*                  sizes = nullptr; // array of sizes for expandable filters. i.e identity can only be 3x3, but gaussian blur can be 3x3, 5x5, 7x7, etc
    unsigned char                   filter_dimension = 0;
    size_t                          name_size;
    bool                            adjustable_strength;

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

        // set expandable based on which filter name it is
        if(strcmp(filter_name, "Identity") == 0) {
            expandable = false;
            custom_weights = false;
        } 
        else if(strcmp(filter_name, "Box blur") == 0) {
            expandable = false;
            custom_weights = false;
        }
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
extern const int* low_pass_filter_data;
extern const int* emboss_filter_data;
extern const int* laplacian_filter_data;
extern const int* motion_blur_filter_data;
extern const int* horizontal_shear_filter_data;
extern const int* vertical_shear_filter_data;
extern const int* sobel_x_filter_data;
extern const int* sobel_y_filter_data;
extern const int* prewitt_x_filter_data;
extern const int* prewitt_y_filter_data;

extern const filter *identity_filter;
extern const filter *edge_filter;
extern const filter *sharpen_filter;
extern const filter *box_blur_filter;
extern const filter *gaussian_blur_filter;
extern const filter *unsharp_mask_filter;
extern const filter *high_pass_filter;
extern const filter *low_pass_filter;
extern const filter *emboss_filter;
extern const filter *laplacian_filter;
extern const filter *motion_blur_filter;
extern const filter *horizontal_shear_filter;
extern const filter *vertical_shear_filter;
extern const filter *sobel_x_filter;
extern const filter *sobel_y_filter;
extern const filter *prewitt_x_filter;
extern const filter *prewitt_y_filter;

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

struct filter_properties get_filter_properties(const char *filter_name);

#endif // __FILTER__H__
