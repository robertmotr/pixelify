#include <stdio.h>
#include <string>
#include <unistd.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION

int filter[] = {
        1,  4,  6,  4,  1,
        4, 16, 24, 16,  4,
        6, 24, 36, 24,  6,
        4, 16, 24, 16,  4,
        1,  4,  6,  4,  1
  };

int main(int argc, char **argv) {
  if(argc != 3) {
    printf("Usage: %s <input> <output>\n", argv[0]);
    return 1;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  int width, height, channels;

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

  printf("Image properties:\n");
  printf("Width: %d\nHeight: %d\nChannels: %d\n", width, height, channels);

  

  return 0;
}
