cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(pixelify CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(SOURCES
    main.cu
    reduce.cu
    kernel.cu 
)

set(HEADERS
    kernel.h
    reduce.h 
    stb_image.h 
    stb_image_write.h
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g --device-debug --diag-suppress=326 -O0")

add_executable(pixelify ${SOURCES} ${HEADERS})

set_target_properties(pixelify PROPERTIES
    CUDA_STANDARD_REQUIRED ON
)
