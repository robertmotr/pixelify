cmake_minimum_required(VERSION 3.28)

project(pixelify LANGUAGES CXX CUDA)

enable_testing()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 61)

add_subdirectory(googletest)
add_subdirectory(cub)

set(SOURCES
    main.cu
    kernel.cu 
    tests.cu
)

set(HEADERS
    kernel.h
    pixel.h
    reduce.h
    stb_image.h 
    stb_image_write.h
)

add_library(pixelify_lib ${SOURCES} ${HEADERS})

set_target_properties(pixelify_lib PROPERTIES
    CUDA_STANDARD_REQUIRED ON
)

add_executable(pixelify main.cu)
target_link_libraries(pixelify PUBLIC pixelify_lib)

add_executable(tests tests.cu)
target_link_libraries(tests PUBLIC pixelify_lib gtest)
include(GoogleTest)
gtest_discover_tests(tests)

set_target_properties(pixelify tests PROPERTIES
    CUDA_STANDARD_REQUIRED ON
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-omit-frame-pointer -fstack-protector-strong -ggdb -G -O0 --diag-suppress=326")