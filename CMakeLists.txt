cmake_minimum_required(VERSION 3.10)

project(pixelify VERSION 1.0)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G, -g, -Wall, -Werror, --device-debug)

include_directories(${CMAKE_SOURCE_DIR})

# CUDA sources
set(CUDA_SOURCES
    kernel.cu
    main.cu
    reduce.cu
)

# C++ sources
set(CPP_SOURCES
    parse.h
    stb_image_write.h
    stb_image.h
)

# Create a CUDA executable
add_executable(pixelify ${CUDA_SOURCES} ${CPP_SOURCES})

# Link libraries if needed
# target_link_libraries(my_cuda_executable some_library)

# Optional: Set build type (Debug or Release)
set(CMAKE_BUILD_TYPE Debug)
