#include <string>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "pixel.h"
#include "kernel.h"
#include "gui.h"

int main(int argc, char **argv) {
  if(argc == 1) {
    // GUI mode
    // call render function
    render_gui_loop();
    return 0;
  }
  else if(argc != 3) {
    printf("Input must be a valid image file, and output is the name of the output file.\n");
    printf("Usage: %s <input> <output>\n", argv[0]);
    return 1;
  }
  else {
  // otherwise cli mode

  // std::string input = argv[2];
  // std::string output = argv[3];

  // int width, height, channels;
  // // get image properties
  // int ok = stbi_info(input.c_str(), &width, &height, &channels);
  // if(ok != 1) {
  //   printf("Failed to get image properties: %s\n", stbi_failure_reason());
  //   return 1;
  // }

  // printf("Image properties:\n");
  // printf("Width: %d\nHeight: %d\nChannels: %d\n", width, height, channels);
  // // load image and get properties
  // /*
  //   points to pixel data consists of *height scanlines of *width pixels,
  //   with each pixel consisting of N interleaved 8-bit components; the first
  //   pixel pointed to is top-left-most in the image. There is no padding between
  //   image scanlines or between pixels, regardless of format.
  // */
  // unsigned char* image_data = stbi_load(input.c_str(), &width, &height, &channels, 0);
  // if (image_data == NULL) {
  //     printf("Failed to load image: %s\n", stbi_failure_reason());
  //     return 1;
  // }

  // unsigned char *image_output = new unsigned char[width * height * channels];

  // if(channels == 3) {
  //   Pixel<3> *pixels_in = raw_image_to_pixel<3>(image_data, width * height);
  //   Pixel<3> *pixels_out = new Pixel<3>[width * height];

  //   run_kernel<3>(blur, 3, pixels_in, pixels_out, width, height);

  //   image_output = pixel_to_raw_image<3>(pixels_out, width * height);
  // }
  // else if(channels == 4) {
  //   Pixel<4> *pixels_in = raw_image_to_pixel<4>(image_data, width * height);
  //   Pixel<4> *pixels_out = new Pixel<4>[width * height];

  //   run_kernel<4>(blur, 3, pixels_in, pixels_out, width, height);

  //   image_output = pixel_to_raw_image<4>(pixels_out, width * height);
  // }
  // else {
  //   // not rgb/rgba so invalid 
  //   printf("Invalid # of channels.\n");
  //   return -1;
  // }

  // stbi_write_png(output.c_str(), width, height, channels, image_output, 0); 

  return 0;
  }
}
