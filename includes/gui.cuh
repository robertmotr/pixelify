#ifndef __UI__H__
#define __UI__H__

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include <exiv2/exiv2.hpp>
#include "kernel.cuh"
#include "pixel.h"
#include "reduce.h"
#include <vector>

#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"
#include <iostream>

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

void display_ui(ImGuiIO& io);

#endif // __UI__H__