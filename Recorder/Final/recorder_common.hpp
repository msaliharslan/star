#ifndef RECORDER_COMMON
#define RECORDER_COMMON

#include <mutex>
#include <iostream>
#include <fstream>
#include <librealsense2/rs.hpp> 

extern bool synch_flag;
extern bool save_flag;

extern std::mutex d435_pipe_mutex;
extern std::mutex t265_pipe_mutex;

const int toas_size = 60;
const int time_difference_threshold = 4;

extern int toas_d435_index;
extern int toas_t265_index;

extern long long toas_d435[toas_size];
extern long long toas_t265[toas_size];


extern unsigned long long initialTimeStamp_d435;
extern unsigned long long initialTimeStamp_t265;


const double depthFramePeriod = 66.66667;// in ms
const double colorFramePeriod = 66.66667;// in ms
const double fisheyeFramePeriod = 33.3333;// in ms


// File related
extern std::ofstream t265_acc;
extern std::ofstream d435_acc;
extern std::ofstream t265_gyro;
extern std::ofstream d435_gyro;
extern std::ofstream t265_pose;

extern std::string fisheye1_folder;
extern std::string fisheye2_folder;
extern std::string color_folder;
extern std::string depth_folder;

// resolution
const int width_color = 848;
const int height_color = 480;

const int width_depth = 1280;
const int height_depth = 720;

void writeVectorToFileBinary(std::ofstream & file, rs2_vector & vector);
void writeQuaternionToFileBinary(std::ofstream & file, rs2_quaternion & quaternion);

unsigned long long getFrameNumber(double framePeriod, double initialTime, double currentTime);

#endif