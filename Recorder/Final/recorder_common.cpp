#include "recorder_common.hpp"

using namespace std;

// Global synch and save flags for callback functions
bool synch_flag = true;
bool save_flag = false;

long long toas_d435[toas_size];
long long toas_t265[toas_size];

int toas_d435_index = 0;
int toas_t265_index = 0;

std::mutex d435_pipe_mutex;
std::mutex t265_pipe_mutex;

// File related
ofstream t265_acc;
ofstream d435_acc;
ofstream t265_gyro;
ofstream d435_gyro;
ofstream t265_pose;

std::string fisheye1_folder;
std::string fisheye2_folder;
std::string color_folder;
std::string depth_folder;



void writeVectorToFileBinary(ofstream & file, rs2_vector & vector){
    
    file.write((const char*) &(vector.x), sizeof(float));
    file.write((const char*) &(vector.y), sizeof(float));
    file.write((const char*) &(vector.z), sizeof(float));

}

void writeQuaternionToFileBinary(ofstream & file, rs2_quaternion & quaternion){
    
    file.write((const char*) &(quaternion.x), sizeof(float));
    file.write((const char*) &(quaternion.y), sizeof(float));
    file.write((const char*) &(quaternion.z), sizeof(float));
    file.write((const char*) &(quaternion.w), sizeof(float));        
}
