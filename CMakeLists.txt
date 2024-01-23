cmake_minimum_required(VERSION 3.28)

project(pixelify LANGUAGES CXX CUDA)
enable_language(CUDA CXX)

include(GoogleTest REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(exiv2 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 61)

set(SOURCES 
    src/main.cu
    src/gui.cu
    src/render.cu
    src/kernel.cu
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

# add_library(pixelify_tests STATIC ${SOURCES})
# target_include_directories(pixelify_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/includes)
# add_executable(tests tests/tests.cu)
# target_link_libraries(tests pixelify_tests gtest pixelify_lib)

# both Debug and Release builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -diag-suppress=326")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -DNDEBUG")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SUFFIX "_debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(SUFFIX "_release")
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()