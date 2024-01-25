#ifndef __FILTER__H__
#define __FILTER__H__

#include <vector>
#include <assert.h>
#include <string.h>
#include <cstring>

typedef struct basic_filter_arrays {
    const int dimension =                                                           3;
    const int identity_filter_data[9] =                                             {0, 0, 0, 0, 1, 0, 0, 0, 0};
    const int box_blur_filter_data[9] =                                             {1, 1, 1, 1, 1, 1, 1, 1, 1};
    const int gaussian_blur_filter_data[9] =                                        {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const int sharpen_filter_data[9] =                                              {0, -1, 0, -1, 5, -1, 0, -1, 0};
    const int edge_detection_filter_data[9] =                                       {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    const int emboss_filter_data[9] =                                               {-2, -1, 0, -1, 1, 1, 0, 1, 2};
    const int sobel_filter_data[9] =                                                {-1, 0, 1, -2, 0, 2, -1, 0, 1}; 
} basic_filter_arrays;

class filter {
private:
    basic_filter_arrays basic_filters;

    int *find_basic_filter(const char *str) {
        if (strcmp(str, "Identity") == 0) {
            return (int *)basic_filters.identity_filter_data;
        } else if (strcmp(str, "Box Blur") == 0) {
            return (int *)basic_filters.box_blur_filter_data;
        } else if (strcmp(str, "Gaussian Blur") == 0) {
            return (int *)basic_filters.gaussian_blur_filter_data;
        } else if (strcmp(str, "Sharpen") == 0) {
            return (int *)basic_filters.sharpen_filter_data;
        } else if (strcmp(str, "Edge Detection") == 0) {
            return (int *)basic_filters.edge_detection_filter_data;
        } else if (strcmp(str, "Emboss") == 0) {
            return (int *)basic_filters.emboss_filter_data;
        } else if (strcmp(str, "Sobel") == 0) {
            return (int *)basic_filters.sobel_filter_data;
        } else {
            return nullptr;
        }
    }

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
        filter_name = new char[size];
        strcpy(filter_name, name);
        this->name_size = size;
    }

    // constructor with name/data specified
    filter(const char* name, unsigned int dimension)
        : filter_dimension(dimension), filter_data(new int[dimension * dimension]) {
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
        if (filter_name != nullptr) {
            delete[] filter_name;
            filter_name = nullptr;
        }
        if (filter_data != nullptr) {
            delete[] filter_data;
            filter_data = nullptr;
        }
        filter_name = new char[other.name_size];
        strcpy(filter_name, other.filter_name);
        filter_dimension = other.filter_dimension;
        filter_data = new int[filter_dimension * filter_dimension];
        memcpy(filter_data, other.filter_data, filter_dimension * filter_dimension * sizeof(int));
        return *this;
    }

    // filter strength on an image is a function of the filters size relative to the images size
    // expand_filter takes in percentage [0, 100], and expands the filter to the largest rectangle that can fit
    // within the image, as a percentage of the image size
    // i.e if percentage = 1, then the filter will be expanded to the largest rectangle that can fit within 1% of the image
    // returns true on success, false on failure
    bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height) {
        if(percentage > 100) {
            return false;
            // since unsigned, percentage/width/height can't be negative
        }

        double desired_area = (percentage / 100.0) * image_width * image_height;

        // Initialize variables to store the best result
        double best_diff = std::numeric_limits<double>::infinity();
        unsigned int best_dimension = 0;

        // Iterate over all possible rectangle sizes
        for (unsigned int dim = 3; dim < std::min(image_width, image_height); dim += 3) {
        
            double current_area = dim * dim;

            // difference between the current area and the desired area
            double diff = std::abs(current_area - desired_area);

            // update best result if the current is closer to desired
            if (diff < best_diff) {
                best_diff = diff;
                best_dimension = dim;
            }
        }

        // it could be possible best_dimension > image_width/image_height
        // in that case scale it back a bit such that it still is within the image and
        // is divisible by 3x3 squares
        while((best_dimension > image_width ||
                best_dimension > image_height ||
                best_dimension % 3 != 0) && 
                best_dimension > 3) {
            best_dimension -= 3;
        }
        assert(best_dimension % 3 == 0 && best_dimension > 0);
        assert(best_dimension <= image_width && best_dimension <= image_height);

        int *copy_data = find_basic_filter(filter_name);

        int *new_filter_data = new int[best_dimension * best_dimension];
        for(int i = 0; i < best_dimension * best_dimension; i += 9) {
            memcpy(new_filter_data + i, copy_data, basic_filters.dimension * basic_filters.dimension * sizeof(int));
        }

        delete[] filter_data;

        filter_dimension = best_dimension;
        filter_data = new_filter_data;

        return true;
    }
};

inline std::vector<filter> get_filter_list() {

    int *IDENTITY_FILTER_DATA = new int[9]{0, 0, 0, 0, 1, 0, 0, 0, 0};
    int *BOX_BLUR_FILTER_DATA = new int[9]{1, 1, 1, 1, 1, 1, 1, 1, 1};
    int *GAUSSIAN_BLUR_FILTER_DATA = new int[9]{1, 2, 1, 2, 4, 2, 1, 2, 1};
    int *SHARPEN_FILTER_DATA = new int[9]{0, -1, 0, -1, 5, -1, 0, -1, 0};
    int *EDGE_DETECTION_FILTER_DATA = new int[9]{-1, -1, -1, -1, 8, -1, -1, -1, -1};
    int *EMBOSS_FILTER_DATA = new int[9]{-2, -1, 0, -1, 1, 1, 0, 1, 2};
    int *SOBEL_FILTER_DATA = new int[9]{-1, 0, 1, -2, 0, 2, -1, 0, 1};

    filter identity_filter("Identity", IDENTITY_FILTER_DATA, 3);
    filter box_blur_filter("Box Blur", BOX_BLUR_FILTER_DATA, 3);
    filter gaussian_blur_filter("Gaussian Blur", GAUSSIAN_BLUR_FILTER_DATA, 3);
    filter sharpen_filter("Sharpen", SHARPEN_FILTER_DATA, 3);
    filter edge_detection_filter("Edge Detection", EDGE_DETECTION_FILTER_DATA, 3);
    filter emboss_filter("Emboss", EMBOSS_FILTER_DATA, 3);
    filter sobel_filter("Sobel", SOBEL_FILTER_DATA, 3);
    filter null = filter();

    std::vector<filter> filter_list = {
        identity_filter,
        box_blur_filter,
        gaussian_blur_filter,
        sharpen_filter,
        edge_detection_filter,
        emboss_filter,
        sobel_filter,
        null
    };

    return filter_list;
}

inline filter create_filter_from_strength(const char *name, unsigned int image_width,
 unsigned int image_height, unsigned char percentage) {

    filter returnable;
    std::vector<filter> filter_list = get_filter_list();
    for (unsigned int i = 0; i < filter_list.size(); i++) {
        if (strcmp(filter_list[i].filter_name, name) == 0) {
            returnable = filter_list[i];
            break;
        }
    }

    if(returnable.expand_filter(percentage, image_width, image_height)) {
        return returnable;
    } else {
        return filter(); // error
    }
}

#endif // __FILTER__H__
