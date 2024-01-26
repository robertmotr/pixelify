#include "gui.h"
#include "kernel.h"
#include "pixel.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"

// normally its kind of pointless to implement a function like this but
// stb_image requires header defines to be in the same file as the implementation
void free_image(unsigned char **image_data) {
    if(image_data != NULL && *image_data != NULL) {
        stbi_image_free(*image_data);
        *image_data = NULL;
    }
}

bool load_texture_from_data(int *out_channels, int *out_width, int *out_height,
                            GLuint *out_texture, unsigned char *image_data) {
    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

        // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    if(!(*out_channels == 3 || *out_channels == 4)) {
        // Handle unsupported channel count
        printf("Unsupported number of channels: %d\n", *out_channels);
        stbi_image_free(image_data);
        return false;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *out_width, *out_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("OpenGL error after glTexImage2D: %x\n", error);
        stbi_image_free(image_data);
        return false;
    }
    *out_texture = image_texture;
    assert(image_texture != 0 && *out_texture != 0);
    return true;
}

// helper function to load an image into a OpenGL texture with common settings
bool load_texture_from_file(const char* filename, GLuint* out_texture, unsigned char **out_raw_image, 
                            int* out_width, int* out_height, int* out_channels) {
    unsigned char* image_data = stbi_load(filename, out_width, out_height, out_channels, 4); 
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return false;
    }
    *out_raw_image = image_data;
    return load_texture_from_data(out_channels, out_width, out_height, out_texture, image_data);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void render_gui_loop() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
    // Create window with graphics context
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);
    GLFWwindow* window = glfwCreateWindow(mode->width, mode->height, "Pixelify", nullptr, nullptr);
    if (window == nullptr)
        return;
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(mode->width, mode->height));

        show_ui(io);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }
        glfwSwapBuffers(window);
    }
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return;
}

bool render_applied_changes(const char* filter_name, struct kernel_args args, int *width, int *height, 
                GLuint *texture_preview, int *channels, unsigned char **image_data_in, unsigned char **image_data_out) {
    assert(image_data_in != NULL);
    assert(*image_data_in != NULL);
    assert(*width > 0);
    assert(*height > 0);
    assert(*channels == 3 || *channels == 4);
    assert(texture_preview != NULL);
    assert(image_data_out != NULL);
    
    if(*channels == 3) {
        Pixel<3> *pixels_in = raw_image_to_pixel<3>(*image_data_in, (*width) * (*height));
        Pixel<3> *pixels_out = new Pixel<3>[(*width) * (*height)];
        run_kernel(filter_name, pixels_in, pixels_out, *width, *height, args);

        if(*image_data_out == NULL) {
            *image_data_out = pixel_to_raw_image<3>(pixels_out, (*width) * (*height));
        }
        else {
            for (unsigned int i = 0; i < (*width) * (*height); i++) {
                for (unsigned int j = 0; j < 3; j++) {
                    if(pixels_in[i].data[j] < 0 || pixels_in[i].data[j] > 255) {
                        printf("Pixel value out of range: %d\n", pixels_in[i].data[j]);
                    }
                    *image_data_out[i * 3 + j] = (unsigned char) pixels_in[i].data[j];
                }
            }
        } 
        
        delete[] pixels_in;
        delete[] pixels_out;
        if(load_texture_from_data(channels, width, height, texture_preview, *image_data_out)) {
            return true;
        }
    }
    else if(*channels == 4) {
        Pixel<4> *pixels_in = raw_image_to_pixel<4>(*image_data_in, (*width) * (*height));
        Pixel<4> *pixels_out = new Pixel<4>[(*width) * (*height)];
        run_kernel(filter_name, pixels_in, pixels_out, *width, *height, args);

        if(*image_data_out == NULL) {
            *image_data_out = pixel_to_raw_image<4>(pixels_out, (*width) * (*height));
        }
        else {
            for (unsigned int i = 0; i < (*width) * (*height); i++) {
                for (unsigned int j = 0; j < 4; j++) {
                    if(pixels_in[i].data[j] < 0 || pixels_in[i].data[j] > 255) {
                        printf("Pixel value out of range: %d\n", pixels_in[i].data[j]);
                    }
                    *image_data_out[i * 4 + j] = (unsigned char) pixels_in[i].data[j];
                }
            }
        }
        
        delete[] pixels_in;
        delete[] pixels_out;
        if(load_texture_from_data(channels, width, height, texture_preview, *image_data_out)) {
            return true;
        }
    }
    return false;
}