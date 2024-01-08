cmake_minimum_required(VERSION 3.28)

project(pixelify LANGUAGES CXX CUDA)

enable_testing()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 61)

set(IMGUI_DIR external/imgui)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-omit-frame-pointer -fstack-protector-strong -ggdb -G -O0 --diag-suppress=326")

