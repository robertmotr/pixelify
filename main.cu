#include <stdio.h>
#include <string>
#include <unistd.h>
#include <libpng/png.h>
#include <setjmp.h>

png_byte* read_png_file(const char* file_name, int* width, int* height) {

    return pixels;
}

int main(int argc, char **argv) {
  if(argc != 3) {
    printf("Usage: %s <input> <output>\n", argv[0]);
    return 1;
  }

  std::string input = argv[1];
  std::string output = argv[2];

  int filter[] = {
        1,  4,  6,  4,  1,
        4, 16, 24, 16,  4,
        6, 24, 36, 24,  6,
        4, 16, 24, 16,  4,
        1,  4,  6,  4,  1
  };


  int width, height;
  png_byte *pixels = read_png_file(input.c_str(), &width, &height);


  return 0;
}
