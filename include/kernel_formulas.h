#ifndef KERNEL_FORMULAS_H
#define KERNEL_FORMULAS_H

#include "filters.h"
#include "filter_impl.h"
#include <unordered_map>
#include <string>

using kernel_formula_fn = float(*)(int i, int j, char strength, unsigned char dimension);
using kernel_formula_map = std::unordered_map<std::string, kernel_formula_fn>;

extern kernel_formula_map* kernel_formulas;

float edge(int i, int j, char strength, unsigned char dimension);
float sharpen(int i, int j, char strength, unsigned char dimension);
float box_blur(int i, int j, char strength, unsigned char dimension);
float gaussian_blur(int i, int j, char strength, unsigned char dimension);
float unsharp_mask(int i, int j, char strength, unsigned char dimension);
float emboss(int i, int j, char strength, unsigned char dimension);
float laplacian(int i, int j, char strength, unsigned char dimension);
float horizontal_shear(int i, int j, char strength, unsigned char dimension);
float vertical_shear(int i, int j, char strength, unsigned char dimension);
// float flip_horizontal(int i, int j, char strength, unsigned char dimension);
// float flip_vertical(int i, int j, char strength, unsigned char dimension);

// kronecker delta
float delta(int i, int j);

void init_kernel_formulas();

#endif