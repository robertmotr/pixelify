#include "gui.h"

void display_ui(const GLFWvidmode *mode) {

    ImGui::Begin("Workshop", nullptr, ImGuiWindowFlags_NoResize
     | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    ImVec2 work_pos = ImGui::GetMainViewport()->WorkPos;
    ImVec2 work_area = ImVec2(work_pos.x + ImGui::GetMainViewport()->WorkSize.x, work_pos.y + 
                                ImGui::GetMainViewport()->WorkSize.y);

    ImVec2 main_panel_size = ImVec2(2 * work_area.x / 3, work_area.y - 30);
    ImVec2 side_panel_size = ImVec2(work_area.x / 3 - 15, work_area.y / 2 - 20);

    ImGui::BeginChild("Main Panel", main_panel_size, true);
    ImGui::Text("Main Panel");
    ImGui::EndChild();

    // Adjust the position of the first side panel
    ImVec2 side_panel1_pos = ImVec2(main_panel_size.x, work_area.y + 10);
    ImGui::SetCursorPos(side_panel1_pos);

    ImGui::BeginChild("Side Panel 1", side_panel_size, true);
    ImGui::Text("Side Panel 1");
    ImGui::EndChild();

    // Adjust the position of the second side panel
    ImVec2 side_panel2_pos = ImVec2(main_panel_size.x + 10, side_panel_size.y);
    ImGui::SetCursorPos(side_panel2_pos);

    ImGui::BeginChild("Side Panel 2", side_panel_size, true);
    ImGui::Text("Side Panel 2");
    ImGui::EndChild();

    ImGui::End();

}