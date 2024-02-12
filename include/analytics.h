#ifndef _ANALYTICS_H
#define _ANALYTICS_H

#include <string>
#include "kernel.cuh"

struct applied_filter {
    float                       time;
    struct filter_args          args;
};

#endif