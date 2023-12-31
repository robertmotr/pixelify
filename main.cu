#include <stdio.h>
#include <string>
#include <unistd.h>
#include "kernel.h"
#include "reduce.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int8_t filter[] = {
  0, 0, 0,
  0, 1, 0,
  0, 0, 0
};

int main(int argc, char **argv) {
  if(argc != 3) {
    printf("Usage: %s <input> <output>\n", argv[0]);
    return 1;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  int width, height, channels;
  // get image properties
  int ok = stbi_info(input.c_str(), &width, &height, &channels);
  if(ok != 1) {
    printf("Failed to get image properties: %s\n", stbi_failure_reason());
    return 1;
  }

  printf("Image properties:\n");
  printf("Width: %d\nHeight: %d\nChannels: %d\n", width, height, channels);
  // load image and get properties
  /*
    points to pixel data consists of *height scanlines of *width pixels,
    with each pixel consisting of N interleaved 8-bit components; the first
    pixel pointed to is top-left-most in the image. There is no padding between
    image scanlines or between pixels, regardless of format.
  */
  unsigned char* image_data = stbi_load(input.c_str(), &width, &height, &channels, 0);
  if (image_data == NULL) {
      printf("Failed to load image: %s\n", stbi_failure_reason());
      return 1;
  }

  unsigned char *image_output = new unsigned char[width * height * channels];

  if(channels == 3) {
    Pixel<3> *pixels_input = new Pixel<3>[width * height];
    for(size_t i = 0; i < width * height; i++) {
      pixels_input[i].data[0] = image_data[i * 3];
      pixels_input[i].data[1] = image_data[i * 3 + 1];
      pixels_input[i].data[2] = image_data[i * 3 + 2];  
    }

    Pixel<3> *pixels_output = new Pixel<3>[width * height];
    run_kernel<3>(filter, 3, pixels_input, pixels_output, width, height);
    
    for(size_t i = 0; i < width * height; i++) {
      image_output[i * 3] = pixels_output[i].data[0];
      image_output[i * 3 + 1] = pixels_output[i].data[1];
      image_output[i * 3 + 2] = pixels_output[i].data[2];
    }
  }
  else if(channels == 4) {
    Pixel<4> *pixels_input = new Pixel<4>[width * height];
    for(size_t i = 0; i < width * height; i++) {
      pixels_input[i].data[0] = image_data[i * 4];
      pixels_input[i].data[1] = image_data[i * 4 + 1];
      pixels_input[i].data[2] = image_data[i * 4 + 2];
      pixels_input[i].data[3] = image_data[i * 4 + 3];
    }
    Pixel<4> *pixels_output = new Pixel<4>[width * height];
    run_kernel<4>(filter, 3, pixels_input, pixels_output, width, height);

    for(size_t i = 0; i < width * height; i++) {
      image_output[i * 4] = pixels_output[i].data[0];
      image_output[i * 4 + 1] = pixels_output[i].data[1];
      image_output[i * 4 + 2] = pixels_output[i].data[2];
      image_output[i * 4 + 3] = pixels_output[i].data[3];
    }
  }
  else {
    // not rgb/rgba so invalid 
    printf("Invalid # of channels.\n");
    return -1;
  }

  stbi_write_png(output.c_str(), width, height, channels, image_output, 0); 

  return 0;
}
