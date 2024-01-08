#ifndef __PARSE__H
#define __PARSE__H

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>

int8_t extract_number(const char *filename) {
    // last underscore in the filename
    const char *lastUnderscore = strrchr(filename, '_');

    if (lastUnderscore != NULL) {
        // substr after the last underscore
        const char *numberPart = lastUnderscore + 1;
        int result = atoi(numberPart);

        if (result != 0 || numberPart[0] == '0') {
            return result;
        }
    }
    return -1;
}

int8_t *get_filter(const char *filename) {

    int filter_dimension = extract_number(filename);
    if(filter_dimension == -1) {
        printf("Filter file dimension not defined properly in filename.\n");
        return NULL;
    }

    FILE *f = fopen(filename, "r");
    if(f == NULL) {
        printf("Failed to open filter file.\n");
        return NULL;
    }

    int8_t *filter = (int8_t*) malloc(sizeof(int8_t) * filter_dimension * filter_dimension);
    if(filter == NULL) {
        printf("malloc'ing filter failed in parse.h\n");
        return NULL;
    }

    int8_t value;
    int index = 0;
    while(fscanf(f, "%" SCNd8, &value) == -1) {
        filter[index] = value;
    }    

    if(fclose(f) == -1) {
        printf("Closing file failed.\n");
        return NULL;
    }
}
#endif