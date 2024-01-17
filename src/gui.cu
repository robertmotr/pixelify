#include "gui.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"

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

    *out_raw_image = image_data;
    *out_texture = image_texture;
    assert(image_texture != 0 && *out_texture != 0);
    return true;
}

inline void display_image(const GLuint& texture, const int& width, const int& height) {
    ImGui::Text("size = %d x %d", width, height);
    static bool use_text_color_for_tint = false;
    ImVec2 pos = ImGui::GetCursorScreenPos();        
    ImVec4 tint_col = use_text_color_for_tint ? ImGui::GetStyleColorVec4(ImGuiCol_Text) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    ImVec4 border_col = ImGui::GetStyleColorVec4(ImGuiCol_Border);

    // Render the original image
    ImGui::Image((void*)(intptr_t)texture, ImVec2(width, height));

    // Check if the mouse is within the bounds of the image
    if (ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + width, pos.y + height))) {
        if (ImGui::BeginTooltip()) {
            float region_sz = 32.0f;
            float region_x = ImGui::GetIO().MousePos.x - pos.x - region_sz * 0.5f;
            float region_y = ImGui::GetIO().MousePos.y - pos.y - region_sz * 0.5f;
            float zoom = 4.0f;

            // Clamp the region within the bounds of the image
            region_x = std::clamp(region_x, 0.0f, static_cast<float>(width - region_sz));
            region_y = std::clamp(region_y, 0.0f, static_cast<float>(height - region_sz));

            ImGui::Text("Min: (%.2f, %.2f)", region_x, region_y);
            ImGui::Text("Max: (%.2f, %.2f)", region_x + region_sz, region_y + region_sz);
            ImVec2 uv0 = ImVec2((region_x) / width, (region_y) / height);
            ImVec2 uv1 = ImVec2((region_x + region_sz) / width, (region_y + region_sz) / height);
            ImGui::Image((void*)(intptr_t)texture, ImVec2(region_sz * zoom, region_sz * zoom), uv0, uv1, tint_col, border_col);
            ImGui::EndTooltip();
        }
    }
}

inline void display_tab_bar(const bool& show_original, const bool& show_preview, const int& width, const int& height, 
                            const GLuint& texture_orig, const GLuint& texture_preview) { 

    if (ImGui::BeginTabBar("tab_bar", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Original image")) {
            if (show_original) {
                display_image(texture_orig, width, height);
            }
            else {
                ImGui::Text("No image loaded. Please select an input file and a filter on the right side panel, and click Apply Changes.");
            }
            ImGui::EndTabItem();
        }
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginTabItem("Preview transformations")) {
            if(show_preview) {
                display_image(texture_preview, width, height);
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
}

std::string generate_exif_string(const Exiv2::ExifData& exifData) {

    if(exifData.empty()) {
        return std::string("No EXIF data found in file\n");
    }

    std::ostringstream result;

    Exiv2::ExifData::const_iterator end = exifData.end();
    for (Exiv2::ExifData::const_iterator i = exifData.begin(); i != end; ++i) {
        const char* tn = i->typeName();
        result << std::setw(44) << std::setfill(' ') << std::left
               << i->key() << " "
               << "0x" << std::setw(4) << std::setfill('0') << std::right
               << std::hex << i->tag() << " "
               << std::setw(9) << std::setfill(' ') << std::left
               << (tn ? tn : "Unknown") << " "
               << std::dec << std::setw(3)
               << std::setfill(' ') << std::right
               << i->count() << "  "
               << std::dec << i->value()
               << "\n";
    }
    return result.str();
}

std::string generateIptcDataString(const Exiv2::IptcData& iptcData) {
    if(iptcData.empty()) {
        return std::string("No IPTC data found in file\n");
    }

    std::ostringstream result;

    auto end = iptcData.end();
    for (auto md = iptcData.begin(); md != end; ++md) {
        result << std::setw(44) << std::setfill(' ') << std::left
               << md->key() << " "
               << "0x" << std::setw(4) << std::setfill('0') << std::right
               << std::hex << md->tag() << " "
               << std::setw(9) << std::setfill(' ') << std::left
               << md->typeName() << " "
               << std::dec << std::setw(3)
               << std::setfill(' ') << std::right
               << md->count() << "  "
               << std::dec << md->value()
               << std::endl;
    }

    return result.str();
}

void display_ui(ImGuiIO& io) {
    // to determine whether which tab is shown
    static bool show_original =             false;
    static bool show_preview =              false;

    // filter options
    static bool normalize =                 false;
    static int filter_strength =            0;
    static int red_strength =               0;
    static int green_strength =             0;
    static int blue_strength =              0;
    static int alpha_strength =             0;
    static int brightness =                 0;
    static ImVec4 tint_colour =             ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    static float blend_factor =             0;

    // image details stuff/rendering stuff
    static char input[256] =                "";
    static char output[256] =               "";
    static int width, height, channels;
    static unsigned char *image_data =      NULL;
    static GLuint texture_orig =            0;
    static GLuint texture_preview =         0;
    static Exiv2::Image::UniquePtr          image;
    static std::string                      exif_data_str;
    static std::string                      iptc_data_str;

    ImGui::Begin("Workshop", nullptr, ImGuiWindowFlags_NoResize
     | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_HorizontalScrollbar
     | ImGuiWindowFlags_AlwaysHorizontalScrollbar);

    ImVec2 main_panel_size = ImVec2(2 * ImGui::GetContentRegionAvail().x / 3,
                                     ImGui::GetContentRegionAvail().y - 75);
    ImVec2 side_panel_1_size = ImVec2(ImGui::GetContentRegionAvail().x / 3,
                                     (2 * ImGui::GetContentRegionAvail().y - 80) / 3);
    ImVec2 side_panel_2_size = ImVec2(ImGui::GetContentRegionAvail().x / 3,
                                     (ImGui::GetContentRegionAvail().y - 80) / 3 - 22);

    ImGui::SetWindowSize(main_panel_size);
    ImVec2 parent_cursor_start = ImGui::GetCursorPos();
    ImGui::BeginChild("Main panel", main_panel_size, ImGuiChildFlags_None, ImGuiWindowFlags_AlwaysHorizontalScrollbar);
    ImGui::SetNextItemWidth(200.0f);
    display_tab_bar(show_original, show_preview, width, height, texture_orig, texture_preview);
    ImGui::EndChild();
    ImGui::SetCursorPos(ImVec2(main_panel_size.x + 10, parent_cursor_start.y));
    ImGui::BeginChild("Side panel 1", side_panel_1_size, true);
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
    ImGui::SameLine();
    // process file path image if user clicks button
    if(ImGui::Button("Open input file")) {
        if(load_texture_from_file(input, &texture_orig, &image_data, &width, &height, &channels) == false) {
            ImGui::OpenPopup("Error loading image");
            show_original = false;
        } else {
            image = Exiv2::ImageFactory::open(input);
            assert(image.get() != 0);
            image->readMetadata();
            Exiv2::ExifData& exifdata = image->exifData();
            exif_data_str = generate_exif_string(exifdata);

            Exiv2::IptcData& iptcdata = image->iptcData();
            iptc_data_str = generateIptcDataString(iptcdata);

            show_original = true;
        }
    }
    ImGui::SameLine();
    if(ImGui::Button("Clear original image")) {
        show_original = false;
        if(image_data != NULL) {
            stbi_image_free(image_data);
            image_data = NULL;
        }
        if(texture_orig != 0) {
            glDeleteTextures(1, &texture_orig);
            texture_orig = 0;
        }
        if(texture_preview != 0) {
            glDeleteTextures(1, &texture_preview);
            texture_preview = 0;
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
    ImGui::SameLine(); 
    ImGui::Spacing();
    ImGui::Spacing();
    // Using the generic BeginCombo() API, you have full control over how to display the combo contents.
    // (your selection data could be an index, a pointer to the object, an id for the object, a flag intrusively
    // stored in the object itself, etc.)
    static ImGuiComboFlags flags = 0;
    static int item_current_idx = 0; // Here we store our selection data as an index.
    const char* combo_preview_value = filter_list[item_current_idx].filter_name.c_str();  // Pass in the preview value visible before opening the combo (it could be anything)
    if (ImGui::BeginCombo("Select filter", combo_preview_value, flags)) {
        for (int n = 0; n < filter_list.size(); n++) {
            const bool is_selected = (item_current_idx == n);
            if (ImGui::Selectable(filter_list[n].filter_name.c_str(), is_selected))
                item_current_idx = n;
            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::SliderInt("Filter strength (0 to 100%)", &filter_strength, 0, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();
    ImGui::SliderInt("Shift reds (-100 to 100%)", &red_strength, -100, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();
    ImGui::SliderInt("Shift blues (-100 to 100%)", &blue_strength, -100, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();
    ImGui::SliderInt("Shift greens (-100 to 100%)", &green_strength, -100, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();
    ImGui::SliderInt("Shift alphas (-100 to 100%)", &alpha_strength, -100, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();
    ImGui::SliderInt("Brightness (-100 to 100%)", &brightness, -100, 100, "%d%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::Spacing();

    ImGui::Checkbox("Normalize image", &normalize);

    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::Text("Tint colour of image");
    ImGui::Spacing();
    float w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.y) * 0.40f;
    ImGui::SetNextItemWidth(w);
    ImGui::ColorPicker4("##tint1", (float*)&tint_colour, ImGuiColorEditFlags_AlphaBar |
        ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_DisplayHex | ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_DisplayHSV);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(w);
    ImGui::ColorPicker3("##tint2", (float*)&tint_colour,  
        ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoSidePreview);
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::SliderFloat("Tint strength (0 to 100%)", &blend_factor, 0.0f, 1.0f, "blend factor = %.3f", ImGuiSliderFlags_AlwaysClamp);

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    if(ImGui::Button("Apply changes")) {
        if(show_original) {
            // nothing for now
        }
    }

    ImGui::EndChild();

    ImGui::SetCursorPos(ImVec2(main_panel_size.x + 10, parent_cursor_start.y + side_panel_1_size.y));
    ImGui::BeginChild("Side panel 2", side_panel_2_size, true);
    ImGui::Text("Image details: ");
    ImGui::Spacing();
    if(show_original) {
        ImGui::Text("Width: %d", width);
        ImGui::Text("Height: %d", height);
        ImGui::Text("Channels: %d", channels);
        ImGui::Text("File size: %d bytes", width * height * channels);
        ImGui::Text("File path: %s", input);
        ImGui::Text("EXIF data: ");
        ImGui::Text("%s", exif_data_str.c_str());
        ImGui::Text("IPTC data: ");
        ImGui::Text("%s", iptc_data_str.c_str());
    }
    else {
        ImGui::Text("Please load an image to view details.");
    }
    ImGui::EndChild();
    ImGui::End();
}