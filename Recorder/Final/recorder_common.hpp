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


void writeVectorToFileBinary(std::ofstream & file, rs2_vector & vector);
void writeQuaternionToFileBinary(std::ofstream & file, rs2_quaternion & quaternion);

#endif