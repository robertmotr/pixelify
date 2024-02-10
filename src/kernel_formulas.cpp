#include "kernel_formulas.h"
#include <math.h>

kernel_formula_map *kernel_formulas;

// kronecker delta
float delta(int i, int j) {
    return (i == j) ? 1.0f : 0.0f;
}

float edge(int i, int j, char strength, unsigned char dimension) {
    return (-1)^(i + j);
}

float sharpen(int i, int j, char strength, unsigned char dimension) {
    // check if (i, j) is center value
    if(i == (dimension - 1) / 2 && j == (dimension - 1) / 2) {
        return (5 + (float) strength / 100.0f);
    }
    // corner values get 0
    if((i == 0 || i == dimension - 1) && (j == 0 || j == dimension - 1)) {
        return 0.0f;
    }
    // edge values get -1
    if(i == 0 || i == dimension - 1 || j == 0 || j == dimension - 1) {
        return -1.0f;
    }
    // return -floatinf as error (shouldnt reach here)
    return -std::numeric_limits<float>::infinity();
}

float box_blur(int i, int j, char strength, unsigned char dimension) {
    return ((float)strength / 10.0f) * 1.0f / (dimension * dimension);
}

float gaussian_blur(int i, int j, char strength, unsigned char dimension) {
    double sigma = static_cast<double>(dimension) / (strength / 10.0f);
    double exponent = -((pow(i - (dimension - 1) / 2, 2) + pow(j - (dimension - 1) / 2, 2)) / (2 * pow(sigma, 2)));
    double value = (1 / (2 * M_PI * pow(sigma, 2))) * exp(exponent);
    return static_cast<float>(value);
}

float unsharp_mask(int i, int j, char strength, unsigned char dimension) {
    float s_factor = -1.0;
    if(i == (dimension - 1) / 2 && j == (dimension - 1) / 2) {
        s_factor = (dimension * dimension - 1) * (float)strength / 5.0f + 1;
    }
    return s_factor;
}

float emboss(int i, int j, char strength, unsigned char dimension) {
    float embossKernel[3][3] = {
        {-1, -1, 0},
        {-1,  1, 1},
        { 0,  1, 1}
    };

    float embossedPixel = 0.0;
    for (int ki = 0; ki < dimension; ++ki) {
        for (int kj = 0; kj < dimension; ++kj) {
            if(ki == i && kj == j) {
                embossedPixel = embossKernel[ki][kj] * (float)strength / 10.0f;
            }
        }
    }

    return embossedPixel;
}

float laplacian(int i, int j, char strength, unsigned char dimension) {
    return (-4 * (float)strength / 10.0f) + delta(i - 1, j) + delta(i + 1, j) + delta(i, j - 1) + delta(i, j + 1);
}

float horizontal_shear(int i, int j, char strength, unsigned char dimension) {

    float h_shear[3][3] = {
        1, 1, 0,
        0, 1, 0,
        0, 0, 1
    };

    if(i == 0 && j == 1) {
        return h_shear[i][j] * (float)strength / 100.0f;
    }
    else {
        return h_shear[i][j];
    }
}

float vertical_shear(int i, int j, char strength, unsigned char dimension) {

    float v_shear[3][3] = {
        1, 0, 0,
        1, 1, 0,
        0, 0, 1
    };

    if(i == 1 && j == 0) {
        return v_shear[i][j] * (float)strength / 100.0f;
    }
    else {
        return v_shear[i][j];
    }
}

void init_kernel_formulas() {

    kernel_formulas = new kernel_formula_map();
    kernel_formulas->emplace("Edge", edge);
    kernel_formulas->emplace("Sharpen", sharpen);
    kernel_formulas->emplace("Box Blur", box_blur);
    kernel_formulas->emplace("Gaussian Blur", gaussian_blur);
    kernel_formulas->emplace("Unsharp Mask", unsharp_mask);
    kernel_formulas->emplace("Emboss", emboss);
    kernel_formulas->emplace("Laplacian", laplacian);
    kernel_formulas->emplace("Horizontal Shear", horizontal_shear);
    kernel_formulas->emplace("Vertical Shear", vertical_shear);
}