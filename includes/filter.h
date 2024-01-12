#ifndef __FILTER__H__
#define __FILTER__H__

#include <string>
#include <vector>

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
};

// list of pre-defined filters here
const int IDENTITY_FILTER_DATA[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
const int BOX_BLUR_FILTER_DATA[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
const int GAUSSIAN_BLUR_FILTER_DATA[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
const int SHARPEN_FILTER_DATA[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
const int EDGE_DETECTION_FILTER_DATA[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
const int EMBOSS_FILTER_DATA[9] = {-2, -1, 0, -1, 1, 1, 0, 1, 2};

const filter IDENTITY_FILTER("IDENTITY", (int *)IDENTITY_FILTER_DATA, 3);
const filter BOX_BLUR_FILTER("BOX_BLUR", (int *)BOX_BLUR_FILTER_DATA, 3);
const filter GAUSSIAN_BLUR_FILTER("GAUSSIAN_BLUR", (int *)GAUSSIAN_BLUR_FILTER_DATA, 3);
const filter SHARPEN_FILTER("SHARPEN", (int *)SHARPEN_FILTER_DATA, 3);
const filter EDGE_DETECTION_FILTER("EDGE_DETECTION", (int *)EDGE_DETECTION_FILTER_DATA, 3);
const filter EMBOSS_FILTER("EMBOSS", (int *)EMBOSS_FILTER_DATA, 3);

#endif // __FILTER__H__
