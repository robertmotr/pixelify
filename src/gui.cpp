#include "gui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

// Simple helper function to load an image into a OpenGL texture with common settings
bool load_texture_from_file(const char* filename, GLuint* out_texture, unsigned char **out_raw_image, 
                            int* out_width, int* out_height, int* out_channels) {
    unsigned char* image_data = stbi_load(filename, out_width, out_height, out_channels, 4); 
    if (image_data == NULL) {
        printf("Failed to load image: %s\n", stbi_failure_reason());
        return false;
    }
    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    GLenum format;
    if (*out_channels == 3)
        format = GL_RGB;
    else if (*out_channels == 4)
        format = GL_RGBA;
    else {
        // Handle unsupported channel count
        printf("Unsupported number of channels: %d\n", *out_channels);
        stbi_image_free(image_data);
        return false;
    }
    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif

    glTexImage2D(GL_TEXTURE_2D, 0, format, *out_width, *out_height, 0, format, GL_UNSIGNED_BYTE, image_data);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("OpenGL error after glTexImage2D: %x\n", error);
        stbi_image_free(image_data);
        return false;
    }

    *out_raw_image = image_data;
    *out_texture = image_texture;
    assert(image_texture != 0 && *out_texture != 0);
    return true;
}


void display_ui(const GLFWvidmode *mode) {

    static int width, height, channels;
    static char input[256] =                "";
    static char output[256] =               "";
    static char filter[256] =               "";
    static bool show_original =             false;
    static bool show_preview =              false;
    static unsigned char *image_data =      NULL;
    static GLuint texture =                 0;

    ImGui::Begin("Workshop", nullptr, ImGuiWindowFlags_NoResize
     | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    ImVec2 main_panel_size = ImVec2(2 * ImGui::GetContentRegionAvail().x / 3,
                                     ImGui::GetContentRegionAvail().y - 75);
    ImVec2 side_panel_size = ImVec2(ImGui::GetContentRegionAvail().x / 3,
                                     (ImGui::GetContentRegionAvail().y - 80) / 2);

    ImGui::SetWindowSize(main_panel_size);
    ImVec2 parent_cursor_start = ImGui::GetCursorPos();
    ImGui::BeginChild("Main panel", main_panel_size, true);
    ImGuiTabBarFlags tab_bar_flags = ImGuiTabBarFlags_None;
    ImGui::SetNextItemWidth(200.0f);
    if (ImGui::BeginTabBar("tab_bar", tab_bar_flags))
    {
        if (ImGui::BeginTabItem("Original image"))
        {
            if(show_original) {
                ImGui::Text("pointer = %p", texture);
                ImGui::Text("size = %d x %d", width, height);
                ImGui::Image((void*)(intptr_t)&texture, ImVec2(width, height));
            } else {
                ImGui::Text("No image loaded. Please select an input file on the right side panel, and click Load Image.");
            }
            ImGui::EndTabItem();
        }
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginTabItem("Preview image"))
        {
            if(show_preview) {
                
            } else {
                ImGui::Text("No image loaded. Please select an input file and a filter on the right side panel, and click Apply Changes.");
            }

            ImGui::EndTabItem();
        }
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginTabItem("Settings")) {
            ImGui::Text("TODO: add settings");   // TODO
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    ImGui::SetCursorPos(ImVec2(main_panel_size.x + 10, parent_cursor_start.y));
    ImGui::BeginChild("Side panel 1", side_panel_size, true);
    ImGui::InputTextWithHint("Input file path", "Absolute path of your input image", input, IM_ARRAYSIZE(input));

    ImGui::Spacing();

    if (ImGui::Button("Select input file")) {
        
        IGFD::FileDialogConfig config;
        config.sidePaneWidth = 300.0f;
        config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".png, .jpg", config);
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string file_path_name = ImGuiFileDialog::Instance()->GetFilePathName();
            std::string file_path = ImGuiFileDialog::Instance()->GetCurrentPath();
            sprintf(input, "%s", file_path_name.c_str());
        }
        ImGuiFileDialog::Instance()->Close();
    }
    // process file path image if user clicks button
    if(ImGui::Button("Open input file")) {
        if(load_texture_from_file(input, &texture, &image_data, &width, &height, &channels) == false) {
            ImGui::OpenPopup("Error loading image");
            show_original = false;
        } else {
            show_original = true;
        }
    }
    if(ImGui::BeginPopup("Error loading image")) {
        ImGui::Text("Error loading image, select a valid path");
        if(ImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::InputTextWithHint("Output file path", "Absolute path for your output image", output, IM_ARRAYSIZE(output));
    ImGui::EndChild();

    ImGui::SetCursorPos(ImVec2(main_panel_size.x + 10, parent_cursor_start.y + side_panel_size.y + 5));
    ImGui::BeginChild("Side panel 2", side_panel_size, true);
    ImGui::Text("Side panel 2");
    ImGui::EndChild();
    
    ImGui::End();

}