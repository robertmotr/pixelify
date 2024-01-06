cmake_minimum_required(VERSION 3.28)

project(pixelify LANGUAGES CXX CUDA)

enable_testing()
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_ARCHITECTURES 61)

add_subdirectory(googletest)
add_subdirectory(cub)

set(IMGUI_DIR imgui)
include_directories(${IMGUI_DIR})

# Add the ImGui source files
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
)

set(SOURCES
    main.cu
    kernel.cu 
    tests.cu
    ${IMGUI_SOURCES}
)

set(HEADERS
    kernel.h
    pixel.h
    reduce.h
    stb_image.h 
    stb_image_write.h
    ${IMGUI_DIR}/imgui.h
    ${IMGUI_DIR}/imgui_internal.h
    ${IMGUI_DIR}/imconfig.h
    ${IMGUIR_DIR}/backends/imgui_impl_glfw.h
)

add_library(pixelify_lib ${SOURCES})
target_include_directories(pixelify_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(pixelify_lib PROPERTIES
    CUDA_STANDARD_REQUIRED ON
)

add_executable(pixelify main.cu)
target_link_libraries(pixelify PUBLIC pixelify_lib)
target_include_directories(pixelify PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}) 

add_executable(tests tests.cu)
target_link_libraries(tests PUBLIC pixelify_lib gtest)
include(GoogleTest)
gtest_discover_tests(tests)

set_target_properties(pixelify tests PROPERTIES
    CUDA_STANDARD_REQUIRED ON
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-omit-frame-pointer -fstack-protector-strong -ggdb -G -O0 --diag-suppress=326")