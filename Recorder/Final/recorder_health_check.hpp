#ifndef RECORDER_HEALTH_CHECK
#define RECORDER_HEALTH_CHECK

#include <librealsense2/rs.hpp>

extern int level_index;

void initHealthCheck(int level);
void health_check_callback_d435(rs2::frame frame);
void health_check_callback_t265(rs2::frame frame);

#endif