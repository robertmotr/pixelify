#include <stdio.h>
#include <string>
#include <unistd.h>

int main(int argc, char **argv) {
  if(argc != 4) {
    printf("Usage: %s <filter> <input> <output>\n", argv[0]);
    return 1;
  }

  std::string filter = argv[1];
  std::string input = argv[2];
  std::string output = argv[3];

  


}
