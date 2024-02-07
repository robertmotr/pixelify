#include "gui.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include "filters.h"
#include "filter_impl.h"
#include "kernel_formulas.h"

inline void display_image(const GLuint& texture, const int& width, const int& height, const unsigned char *image_data) {

    ImGui::Text("size = %d x %d", width, height);
    ImVec2 pos = ImGui::GetCursorScreenPos();   

    // get io
    ImGuiIO& io = ImGui::GetIO();

    // Check if the mouse is within the bounds of the image
    if (ImGui::IsMouseHoveringRect(pos, ImVec2(pos.x + width, pos.y + height))) {

        ImGui::BeginTooltip();
        ImGui::Text("Ctrl + Scroll to zoom in/out");
        ImGui::EndTooltip();

        ImVec2 mouse_pos = ImGui::GetMousePos();
        
        // Correct the UV coordinate calculation
        ImVec2 mouse_uv_coords = ImVec2((mouse_pos.x - pos.x) / width, (mouse_pos.y - pos.y) / height);
        ImVec2 displayed_texture_size = ImGui::GetItemRectSize();
        // Call ImageInspect::inspect with corrected UV coordinates
        ImageInspect::inspect(width, height, image_data, mouse_uv_coords, displayed_texture_size);
    }

    ImGui::Image((void*)(intptr_t)texture, ImVec2(width, height));
}


inline void display_tab_bar(bool& original_loaded, bool& preview_loaded, const int& width, const int& height, 
                            const GLuint& texture_orig, const GLuint& texture_preview, const unsigned char *image_orig, 
                            const unsigned char *image_preview) { 

    if (ImGui::BeginTabBar("tab_bar", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Original image")) {
            if(original_loaded) {
                display_image(texture_orig, width, height, image_orig);
            }
            else {
                ImGui::Text("No image loaded. Please select an input file and a filter on the right side panel, and click Apply Changes.");
            }
            ImGui::EndTabItem();
        }
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginTabItem("Preview transformations")) {
            if(preview_loaded) {
                display_image(texture_preview, width, height, image_preview);
            }
            else {
                ImGui::Text("No image loaded. Please select an input file and a filter on the right side panel, and click Apply Changes.");
            }
            ImGui::EndTabItem();
        }
        ImGui::SetNextItemWidth(200.0f);
        if (ImGui::BeginTabItem("Settings")) {
            ImGui::Text("TODO: add settings");   // TODO
            ImGui::Text("But, to be honest, theres not much to add here.");
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

std::string generate_iptc_string(const Exiv2::IptcData& iptcData) {
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

void show_ui(ImGuiIO& io) {
    // to determine whether which tab is shown
    static bool show_original =                 false;
    static bool show_preview =                  false;
    static bool show_tint =                     false;

    // filter options
    static bool normalize =                     false;
    static int filter_size =                    3;
    static int filter_strength =                0;
    static int red_strength =                   0;
    static int green_strength =                 0;
    static int blue_strength =                  0;
    static int alpha_strength =                 0;
    static int brightness =                     0;
    static ImVec4 tint_colour =                 ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    static float blend_factor =                 0;

    // image details stuff
    static char input[256] =                    "";
    static char output[256] =                   "";
    static Exiv2::Image::UniquePtr              image;
    static std::string                          exif_data_str;
    static std::string                          iptc_data_str;

    // rendering stuff

    static int width, height, channels;
    static unsigned char *image_data =          NULL;
    static unsigned char *image_data_out =      NULL;
    static GLuint texture_orig =                0;
    static GLuint texture_preview =             0;
    static const filter** filters =             init_filters();
    static int current_filter_dropdown_idx =    0;
    static ImGuiComboFlags flags =              0;
    // cast filters to non-const

    static filter* selected_filter =            const_cast<filter*>(filters[current_filter_dropdown_idx]);   

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
    display_tab_bar(show_original, show_preview, width, height, texture_orig, texture_preview, image_data, image_data_out);
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
            iptc_data_str = generate_iptc_string(iptcdata);

            show_original = true;
        }
    }
    ImGui::SameLine();
    if(ImGui::Button("Clear original image")) {
        show_original = false;
        if(image_data != NULL) {
            free_image(&image_data);
        }
        if(texture_orig != 0) {
            glDeleteTextures(1, &texture_orig);
            texture_orig = 0;
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
    ImGui::InputTextWithHint("##output", "Absolute path for your output image", output, IM_ARRAYSIZE(output));
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::Spacing();
    // Using the generic BeginCombo() API, you have full control over how to display the combo contents.
    // (your selection data could be an index, a pointer to the object, an id for the object, a flag intrusively
    // stored in the object itself, etc.)
    const char* combo_preview_value = filters[current_filter_dropdown_idx]->filter_name;  // Pass in the preview value visible before opening the combo (it could be anything)
    if (ImGui::BeginCombo("Select filter", combo_preview_value, flags)) {
        for (int n = 0; n < BASIC_FILTER_SIZE; n++) {
            const bool is_selected = (current_filter_dropdown_idx == n);
            if (ImGui::Selectable(filters[n]->filter_name, is_selected))
                current_filter_dropdown_idx = n;
                selected_filter = const_cast<filter*>(filters[current_filter_dropdown_idx]);
            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if(selected_filter->properties->lower_bound_strength == selected_filter->properties->upper_bound_strength) {
        // just print text instead of slider
        ImGui::Text("Filter strength: %d (not adjustable)", filter_strength);
    }
    else {
        ImGui::SliderInt("Filter strength", &filter_strength, 
                    selected_filter->properties->lower_bound_strength, selected_filter->properties->upper_bound_strength, "%d", ImGuiSliderFlags_AlwaysClamp);
    }
    ImGui::Spacing();
    // check if selected filter is adjustable by size, if so then allow user to iterate through sizes
    if(selected_filter->properties->expandable_size) {
        int min_size = selected_filter->properties->sizes_avail[0];
        int max_size = selected_filter->properties->sizes_avail[selected_filter->properties->num_sizes_avail - 1];

        // Display a single slider for the range of values in sizes_avail
        ImGui::SliderInt("Filter size", &filter_size, min_size, max_size);

        // clamp filter_size such that it is odd i.e if user selects even then set it to odd
        if(filter_size % 2 == 0) {
            filter_size++;
            if(filter_size > max_size) {
                filter_size = max_size;
            }
            ImGui::Text("Filter size must be odd, setting to %d", filter_size);
        }
    }
    else {
        filter_size = 3;
        ImGui::Text("Filter size: %d (not adjustable)", filter_size);
    }
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

    ImGui::Checkbox("Tint image", &show_tint);
    ImGui::Spacing();
    ImGui::Spacing();
    if(show_tint) {
        ImGui::Text("Tint colour of image");
        ImGui::Spacing();
        float w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.y) * 0.40f;
        ImGui::SetNextItemWidth(w);
        ImGui::ColorPicker4("##tint1", (float*)&tint_colour, ImGuiColorEditFlags_AlphaBar |
            ImGuiColorEditFlags_PickerHueBar | ImGuiColorEditFlags_DisplayHex | ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_DisplayHSV
            | ImGuiColorEditFlags_AlphaPreviewHalf | ImGuiColorEditFlags_HDR);
        
        ImGui::SameLine();
        ImGui::SetNextItemWidth(w);
        ImGui::ColorPicker4("##tint2", (float*)&tint_colour,  
            ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoSidePreview);
        ImGui::Spacing();
        ImGui::Spacing();

        ImGui::SliderFloat("Tint strength (0 to 100%)", &blend_factor, 0.0f, 1.0f, "blend factor = %.3f", ImGuiSliderFlags_AlwaysClamp);

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Spacing();
    }

    if(ImGui::Button("Apply changes")) {
        if(show_original) {
            show_preview = true;
            // pass kernel args to render function
            struct kernel_args extra_args;
            extra_args.red_shift = static_cast<char>(red_strength);
            extra_args.green_shift = static_cast<char>(green_strength);
            extra_args.blue_shift = static_cast<char>(blue_strength);
            extra_args.alpha_shift = static_cast<char>(alpha_strength);
            extra_args.brightness = static_cast<char>(brightness);
        
            extra_args.normalize = normalize;
            extra_args.filter_strength = static_cast<char>(filter_strength);
            extra_args.blend_factor = blend_factor;
            extra_args.tint[0] = static_cast<unsigned char>(tint_colour.x);
            extra_args.tint[1] = static_cast<unsigned char>(tint_colour.y);
            extra_args.tint[2] = static_cast<unsigned char>(tint_colour.z);
            extra_args.tint[3] = static_cast<unsigned char>(tint_colour.w);

            render_applied_changes(selected_filter->filter_name, extra_args, width, height, &texture_preview, channels,
                                    &image_data, &image_data_out, input);

        }
    }
    ImGui::SameLine();
    if(ImGui::Button("Clear all changes")) {
        show_preview = false;
        if(texture_preview != 0) {
            glDeleteTextures(1, &texture_preview);
            texture_preview = 0;
        }
        if(image_data_out != NULL) free_image(&image_data_out);
        if(image_data != NULL) free_image(&image_data);

    }
    ImGui::SameLine();
    if(ImGui::Button("Restore defaults")) {
        filter_strength = 0;
        red_strength = 0;
        green_strength = 0;
        blue_strength = 0;
        alpha_strength = 0;
        brightness = 0;
        normalize = false;
        blend_factor = 0;
        tint_colour = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        show_tint = false;
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