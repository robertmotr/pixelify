#include "kernel_formulas.h"

kernel_formula_map formulas;

int kronecker_delta(int i, int j) {
    return (i == j) ? 1 : 0;
}

float kronecker_delta_f(int i, int j) {
    return (i == j) ? 1.0f : 0.0f;
}

float edge(int i, int j, char strength, unsigned char dimension) {
    return (-1)^(i + j);
}

float sharpen(int i, int j, char strength, unsigned char dimension) {
    return 2 * kronecker_delta_f(i, j) - 1.0f;
}

float box_blur(int i, int j, char strength, unsigned char dimension) {
    return 1.0f / (dimension * dimension);
}

float gaussian_blur(int i, int j, char strength, unsigned char dimension) {
    float sigma = (float)strength;
    float x = (float)i - (float)dimension / 2.0f;
    float y = (float)j - (float)dimension / 2.0f;
    return expf(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
}

float unsharp_mask(int i, int j, char strength, unsigned char dimension) {
    return strength * (kronecker_delta_f(i, 0) * kronecker_delta_f(j, 0) - 1/(dimension * dimension));
}

float high_pass(int i, int j, char strength, unsigned char dimension) {
    return kronecker_delta_f(i, 0) * kronecker_delta_f(j, 0) - 1/(dimension * dimension);
}

float emboss(int i, int j, char strength, unsigned char dimension) {
    return 2 * kronecker_delta_f(i, j) - 1.0f;
}

float laplacian(int i, int j, char strength, unsigned char dimension) {
    return -4 + kronecker_delta_f(i - 1, j) + kronecker_delta_f(i + 1, j) + kronecker_delta_f(i, j - 1) + kronecker_delta_f(i, j + 1);
}

float horizontal_shear(int i, int j, char strength, unsigned char dimension) {
    return kronecker_delta_f(i, 0) * kronecker_delta_f(j, floor(strength * i));
}

float vertical_shear(int i, int j, char strength, unsigned char dimension) {
    return kronecker_delta_f(i, floor(strength * j)) * kronecker_delta_f(j, 0);
}

__attribute__((constructor))
void init_kernel_formulas() {

    formulas = kernel_formula_map();
    formulas["Edge"] = edge;
    formulas["Sharpen"] = sharpen;
    formulas["Box Blur"] = box_blur;
    formulas["Gaussian Blur"] = gaussian_blur;
    formulas["Unsharp Mask"] = unsharp_mask;
    formulas["High Pass"] = high_pass;
    formulas["Emboss"] = emboss;
    formulas["Laplacian"] = laplacian;
    formulas["Horizontal Shear"] = horizontal_shear;
    formulas["Vertical Shear"] = vertical_shear;
    
}