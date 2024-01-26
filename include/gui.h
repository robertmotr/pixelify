#ifndef __GUI__H
#define __GUI__H

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuiFileDialog.h"

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "kernel.h"
#include "pixel.h"
#include "filter.h"

#include <vector>
#include <iostream>
#include <exiv2/exiv2.hpp>

void free_image(unsigned char **image_data);

bool load_texture_from_data(int *out_channels, int *out_width, int *out_height, GLuint *out_texture, unsigned char *image_data);
// Simple helper function to load an image into a OpenGL texture with common settings
bool load_texture_from_file(const char* filename, GLuint* out_texture, unsigned char **out_raw_image, 
                            int* out_width, int* out_height, int* out_channels);

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// main function for rendering the gui through imgui, little to no ui stuff here
void render_gui_loop();

// returns true on success and false on failure
// calls run_kernel to process the original image into preview image using the filter + other changes
bool render_applied_changes(const char* filter_name, struct kernel_args args, int *width, int *height, 
                GLuint *texture_preview, int *channels, unsigned char **image_data_in, unsigned char **image_data_out);

// displays image in the GUI given a gluint texture
inline void display_image(const GLuint& texture, const int& width, const int& height);

// self explanatory
inline void display_tab_bar(const bool& show_original, const bool& show_preview, const int& width, const int& height, 
                            const GLuint& texture_orig, const GLuint& texture_preview);

// generates exif string given an exiv2 exifdata object
std::string generate_exif_string(const Exiv2::ExifData& exifData);

// same thing but for IPTC
std::string generate_iptc_string(const Exiv2::IptcData& iptcData);

// main ui loop for the program
// all ui elements/stuff is here
void show_ui(ImGuiIO& io);

#endif // __GUI__H