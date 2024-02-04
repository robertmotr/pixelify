cmake_minimum_required(VERSION 3.28)

project(pixelify LANGUAGES CXX CUDA)
enable_language(CUDA CXX)

include(GoogleTest REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(exiv2 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(SOURCES 
    src/main.cu
    src/gui.cu
    src/render.cu
    src/kernel.cu
    src/filters.cu
    src/filter_impl.cu
    src/kernel_formulas.cu
)

set(IMGUI_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/misc/cpp/imgui_stdlib.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends/imgui_impl_opengl3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiFileDialog/ImGuiFileDialog.cpp
)

add_library(imgui_lib STATIC ${IMGUI_SOURCES})
target_include_directories(imgui_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)

add_library(pixelify_lib STATIC ${SOURCES})
target_include_directories(pixelify_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cub/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiFileDialog
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
add_executable(pixelify src/main.cu)
target_link_libraries(pixelify PUBLIC pixelify_lib glfw imgui_lib OpenGL::GL exiv2)

# ------ TESTS ------

set(SOURCES_TEST 
    src/kernel.cu 
    src/filters.cu
    src/filter_impl.cu
    src/kernel_formulas.cu
)

set(HEADERS_TEST 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/filters.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_impl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_formulas.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/pixel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/reduce.h 
)

add_library(pixelify_tests STATIC ${SOURCES_TEST})
target_include_directories(pixelify_tests PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/external/cub/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
add_executable(tests tests/tests.cu)
target_link_libraries(tests pixelify_tests gtest)

# both Debug and Release builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3 -O0 -fno-omit-frame-pointer -fstack-protector-all -Wformat -Wformat-security -Wshadow")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -O0 -diag-suppress=326")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use-fast-math -DNDEBUG")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SUFFIX "_debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(SUFFIX "_release")
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()