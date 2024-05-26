cmake_minimum_required(VERSION 3.22)

project(pixelify LANGUAGES CXX CUDA)
include(CTest)
include(GoogleTest REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(exiv2 REQUIRED)
enable_testing()

# set architecture flags
# this depends on your GPU
# see link below
# https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 61)

set(SOURCES 
    src/main.cu
    src/gui.cu
    src/render.cu
    src/kernel.cu
    src/filters.cpp
    src/filter_impl.cpp
    src/kernel_formulas.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect/imgui_tex_inspect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect/backends/tex_inspect_opengl.cpp
)

add_library(imgui_lib ${IMGUI_SOURCES})
target_include_directories(imgui_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiFileDialog
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgInspect 
)
target_compile_options(imgui_lib PRIVATE -DIMGUI_IMPL_OPENGL_LOADER_GLEW)

add_library(pixelify_lib STATIC ${SOURCES})
target_include_directories(pixelify_lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ImGuiFileDialog
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgInspect
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_tex_inspect/backends
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPENGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS}
)

add_executable(pixelify src/main.cu)
target_include_directories(pixelify PRIVATE ${INCLUDE_DIRS})
target_link_libraries(pixelify PUBLIC 
    ${OPENGL_LIBRARIES}
    pixelify_lib 
    glfw 
    imgui_lib 
    OpenGL::GL 
    exiv2 
    GLEW
    ${CMAKE_DL_LIBS}    
)

# ------ TESTS ------

# set(SOURCES_TEST 
#     src/kernel.cu 
#     src/filters.cpp
#     src/filter_impl.cpp
#     src/kernel_formulas.cpp
# )

# set(HEADERS_TEST 
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel.cuh
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/filters.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/filter_impl.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel_formulas.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/pixel.h
# )

# add_library(pixelify_tests STATIC ${SOURCES_TEST})
# target_include_directories(pixelify_tests PUBLIC 
#     ${CMAKE_CURRENT_SOURCE_DIR}/external
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/googletest/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/external/googletest/googlemock/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )
# add_executable(tests tests/tests.cu)
# target_link_libraries(tests pixelify_tests gtest_main gtest)

# both Debug and Release builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_DEBUG -ggdb3 -fno-omit-frame-pointer -fstack-protector-all -Wformat -Wformat-security")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -D_DEBUG -O0 -G -diag-suppress=326")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -fomit-frame-pointer -fno-stack-protector -flto")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -Xptxas -O3 -use_fast_math")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(SUFFIX "_debug")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(SUFFIX "_release")
else()
    message(FATAL_ERROR "Unsupported build type: ${CMAKE_BUILD_TYPE}")
endif()
