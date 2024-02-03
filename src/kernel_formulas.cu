#include "kernel_formulas.h"

int kronecker_delta(int i, int j) {
    return (i == j) ? 1 : 0;
}