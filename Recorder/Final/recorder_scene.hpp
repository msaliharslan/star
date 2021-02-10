#ifndef RECORDER_SCENE
#define RECORDER_SCENE

#include <librealsense2/rs.hpp>
#include <recorder_common.hpp>

void scene_callback_d435(rs2::frame frame);
void scene_callback_t265(rs2::frame frame);

void initScene(std::string name);

#endif
