#ifndef __FILTER__H__
#define __FILTER__H__

#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <string.h>

class filter {
    public:
        std::string filter_name;
        int *filter_data;
        unsigned int filter_dimension;

        filter() {
            filter_name = "";
            filter_dimension = 0;
            filter_data = NULL;
        }

        filter(std::string name, int *data, unsigned int dimension) {
            filter_name = name;
            filter_data = data;
            filter_dimension = dimension;
        }
        filter(std::string name, unsigned int dimension) {
            filter_name = name;
            filter_dimension = dimension;
            filter_data = new int[dimension * dimension];
        }
        filter(std::string name, unsigned int dimension, int value) {
            filter_name = name;
            filter_dimension = dimension;
            filter_data = new int[dimension * dimension];
            for (unsigned int i = 0; i < dimension * dimension; i++) {
                filter_data[i] = value;
            }
        }

        ~filter() {
            delete[] filter_data;
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
        // filter strength on an image is a function of the filters size relative to the images size
        // expand_filter takes in percentage [0, 100], and expands the filter to the largest rectangle that can fit
        // within the image, as a percentage of the image size
        // i.e if percentage = 1, then the filter will be expanded to the largest rectangle that can fit within 1% of the image
        // returns true on success, false on failure
        bool expand_filter(unsigned char percentage, unsigned int image_width, unsigned int image_height) {
            if(percentage > 100 || image_width < 1 || image_height < 1) {
                return false;
            }

            unsigned int max_width = image_width * percentage / 100;
            unsigned int max_height = image_height * percentage / 100;

            if(max_width < 9 || max_height < 9) {
                max_width = std::max(max_width, max_height, 9);
                max_height = max_width;
            }
            // assert max_width and max_height are divisible by 9
            // also assert max_width == max_height
            assert(max_width * max_height % 9 == 0);
            assert(max_width == max_height);

            int *new_filter_data = new int[max_width * max_height];
            for(int i = 0; i < max_width * max_height; i += 9) {
                // memcpy our filter data into the new filter data
                memcpy(new_filter_data + i, filter_data, filter_dimension * filter_dimension * sizeof(int));
            }
            // ideally we'd delete old filter data here, but we assume that old filter data is always allocated on the stack
            filter_data = new_filter_data;
            filter_dimension = max_width;
            return true;
        }
};

// list of pre-defined filters here
const int IDENTITY_FILTER_DATA[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
const int BOX_BLUR_FILTER_DATA[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
const int GAUSSIAN_BLUR_FILTER_DATA[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
const int SHARPEN_FILTER_DATA[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
const int EDGE_DETECTION_FILTER_DATA[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
const int EMBOSS_FILTER_DATA[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

const filter NULL_FILTER();
NULL_FILTER.filter_name = std::string("NULL");
const filter IDENTITY_FILTER("IDENTITY", (int *)IDENTITY_FILTER_DATA, 3);
const filter BOX_BLUR_FILTER("BOX_BLUR", (int *)BOX_BLUR_FILTER_DATA, 3);
const filter GAUSSIAN_BLUR_FILTER("GAUSSIAN_BLUR", (int *)GAUSSIAN_BLUR_FILTER_DATA, 3);
const filter SHARPEN_FILTER("SHARPEN", (int *)SHARPEN_FILTER_DATA, 3);
const filter EDGE_DETECTION_FILTER("EDGE_DETECTION", (int *)EDGE_DETECTION_FILTER_DATA, 3);
const filter EMBOSS_FILTER("EMBOSS", (int *)EMBOSS_FILTER_DATA, 3);

std::vector<filter> filter_list = {
    IDENTITY_FILTER,
    BOX_BLUR_FILTER,
    GAUSSIAN_BLUR_FILTER,
    SHARPEN_FILTER,
    EDGE_DETECTION_FILTER,
    EMBOSS_FILTER,
    NULL_FILTER
};

filter create_filter_from_strength(std::string name, unsigned int image_width, unsigned int image_height, unsigned char percentage) {
    filter f;
    for(auto filter : filter_list) {
        if(filter.filter_name == name) {
            f = filter;
            break;
        }
    }

    if(f.expand_filter(percentage, image_width, image_height)) {
        return f;
    } else {
        return filter(); // error
    }
}

#endif // __FILTER__H__
