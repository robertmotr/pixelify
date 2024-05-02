#include <string>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "pixel.h"
#include "kernel.cuh"
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
    // otherwise cli mode(?)
    // TODO 
  return 0;
  }
}
