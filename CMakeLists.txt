cmake_minimum_required(VERSION 3.28)
set(CMAKE_VERBOSE_MAKEFILE on)

project(pixelify LANGUAGES CXX CUDA)

# Set build type and CUDA architectures
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 61)

# Set additional CUDA flags

# Output binaries to the 'build' directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Define your source and header files
set(SOURCES
    src/gui.cpp
    src/kernel.cu
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/pixel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/reduce.h
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/gui.h
)

# pixelify lib
add_library(pixelify_lib ${SOURCES} ${HEADERS})
target_include_directories(pixelify_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/includes 
    ${CMAKE_CURRENT_SOURCE_DIR}/external 
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(pixelify_lib PUBLIC cub)
set_target_properties(pixelify_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# pixelify executable
add_executable(pixelify src/main.cu)
target_include_directories(pixelify PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)
target_link_libraries(pixelify PUBLIC pixelify_lib)

# tests executable
enable_testing()
add_executable(tests tests/tests.cu)
target_link_libraries(tests PUBLIC gtest pixelify_lib)
set_target_properties(tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/includes)