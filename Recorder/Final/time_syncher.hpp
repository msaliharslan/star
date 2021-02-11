#ifndef TIME_SYNCHER
#define TIME_SYNCHER

#include <librealsense2/rs.hpp> 
#include <iostream>
#include "recorder_calibration.hpp"
#include "recorder_health_check.hpp"
#include "recorder_scene.hpp"
#include "recorder_common.hpp"
#include <chrono>
#include <thread>
#include <mutex>
#include <experimental/random>

bool timeSyncher(rs2::pipeline *&pipe1, void (*callback1)(rs2::frame), rs2::pipeline *&pipe2, void (*callback2)(rs2::frame));

#endif // TIME_SYNCHER