#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <thread>
#include <fstream>
#include <experimental/filesystem>
#include <time.h>
#include <chrono>

using namespace rs2;
using namespace std;

namespace fs = std::experimental::filesystem;

const double deltaT = 3; //seconds
const int numShots = 6; 
const double warmUpTime = 7; //seconds

bool continueRecord = true;

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


int main(int argc, char **argv) try {

    //initialize

        //create folder

    //first generate the new recording directory name
    int currentSaveIndex = 0;

    //find currentSaveIndex
    std::string path = "../../Records";
    for (const auto & entry : fs::directory_iterator(path)){
        cout << entry.path() << endl;
        std::string delimiter1 = "/";
        std::string delimeter2 = "_";
        std::string entryPath = std::string(entry.path());
        std::reverse(entryPath.begin(), entryPath.end());
        std::string fileName = (entryPath.substr(0, entryPath.find(delimiter1)));
        std::reverse(fileName.begin(), fileName.end());
        std::string saveIndexString = fileName.substr(0, fileName.find(delimeter2));
        int index = stoi(saveIndexString);
        if(index > currentSaveIndex)
            currentSaveIndex = index;        
    }

    currentSaveIndex++;
    cout << "currentSaveIndex " << currentSaveIndex << endl;

    time_t rawtime;
    struct tm * timeinfo;

    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime (buffer,80,"_%F_%R\0AA",timeinfo);

    std::stringstream saveFolderName;
    saveFolderName.str("");
    saveFolderName << currentSaveIndex;
    saveFolderName << buffer;
    cout << saveFolderName.str() << endl;


        //create folders

    //create main folder
    fs::path p;
    p = fs::current_path();    
    string command1 = "mkdir ";
    command1 += p;    
    command1 += "/../../Records/"; 
    command1 += saveFolderName.str();
    system(command1.c_str());
    
    //create leftFisheye folder
    string command2 = command1;
    command2 += "/leftFisheye";
    system(command2.c_str());

    //create rightFisheye folder
    string command3 = command1;
    command3 += "/rightFisheye";
    system(command3.c_str());

    //create rgb folder
    string command4 = command1;
    command4 += "/rgb";
    system(command4.c_str());

    //create depth folder
    string command5 = command1;
    command5 += "/depth";
    system(command5.c_str());    

        //init file objects   
    string filesPath = p;
    filesPath += "/../../Records/";
    filesPath += saveFolderName.str();

    //acc

    //init t265_acc.bin

    cout << filesPath << endl;

    ofstream t265_acc;
    t265_acc.open((filesPath + "/t265_acc.bin").c_str(), ios::out | ios::binary);

    //init d435_acc.bin

    ofstream d435_acc;
    d435_acc.open((filesPath + "/d435_acc.bin").c_str(), ios::out | ios::binary);

    //gyro

    //init t265_acc.bin

    ofstream t265_gyro;
    t265_gyro.open((filesPath + "/t265_gyro.bin").c_str(), ios::out | ios::binary);

    //init d435_acc.bin

    ofstream d435_gyro;
    d435_gyro.open((filesPath + "/d435_gyro.bin").c_str(), ios::out | ios::binary);


    //pose

    ofstream t265_pose;
    t265_pose.open((filesPath + "/t265_pose.bin").c_str(), ios::out | ios::binary);    

    rs2::config cfg1, cfg2;
    rs2::device dev;
    rs2::colorizer color_map;
    rs2::context ctx;
    rs2::frameset frames;
    rs2::frame frame;

    // Obtain a list of devices currently present on the system

    bool D435Connected = false;
    bool T265Connected = false;
    auto devices = ctx.query_devices();
    size_t device_count = devices.size();
    if (!device_count){
        cout << "No device is found!!" << endl;
        return EXIT_SUCCESS;        
    }

  
    for (int i = 0; i < device_count; i++){
        string deviceName = devices[i].get_info(RS2_CAMERA_INFO_NAME);
        cout << i + 1 << "\t" << deviceName << endl;
        size_t found = deviceName.find("D435");
        if(found <= deviceName.length()) {
            D435Connected = true;

            devices[i].first<rs2::color_sensor>().set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0); 
            devices[i].first<rs2::depth_stereo_sensor>().set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0); 
            devices[i].first<rs2::motion_sensor>().set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0); 
            
            cfg2.enable_device(devices[i].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
            // cfg2.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
            // cfg2.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
            cfg2.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
            cfg2.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);

        }
        else {
            T265Connected = true;
            cfg1.enable_device(devices[i].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
            cfg1.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
            cfg1.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
            cfg1.enable_stream(RS2_STREAM_POSE);
            cfg1.enable_stream(RS2_STREAM_FISHEYE, 1);
            cfg1.enable_stream(RS2_STREAM_FISHEYE, 2);
        }



    }





    pipeline pipe1(ctx);
    pipeline pipe2(ctx);

    int shotCounter = 1;
    auto startTime = std::chrono::steady_clock::now();


    //T265

    if(T265Connected) {

        pipe1.start(cfg1, [&](rs2::frame frame)
        {

            auto currentTime = chrono::steady_clock::now();
            auto passedTime = chrono::duration_cast<chrono::milliseconds>( currentTime - startTime ).count(); // in milliseconds

            if( passedTime > warmUpTime * 1000 && shotCounter <= numShots){

                if((passedTime - warmUpTime * 1000) / (deltaT * 1000) > shotCounter){
                    cout << "shot taken t265" << endl;
                    shotCounter ++;
                }

                // Cast the frame to appropriate object type
                auto fset = frame.as<rs2::frameset>();
                auto motion = frame.as<rs2::motion_frame>();
                auto fisheye1 = fset.get_fisheye_frame(1);
                auto fisheye2 = fset.get_fisheye_frame(2);
                auto pose = frame.as<rs2::pose_frame>();
                

                // If casting succeeded and the arrived frame is from gyro stream
                if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
                {
                    // Get the timestamp of the current frame
                    // ts2 = motion.get_timestamp();
                    // Get gyro measures
                    rs2_vector gyro_data = motion.get_motion_data();

                    double timeStamp = motion.get_timestamp();
                    t265_gyro.write((const char*)(&timeStamp), sizeof(double));            

                    writeVectorToFileBinary(t265_gyro, gyro_data);

                }

                // If casting succeeded and the arrived frame is from accelerometer stream
                if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
                {

                    // Get accelerometer measures
                    rs2_vector accel_data = motion.get_motion_data();

                    double timeStamp = motion.get_timestamp();
                    t265_acc.write((const char*)(&timeStamp), sizeof(double));            

                    writeVectorToFileBinary(t265_acc, accel_data);

                }

                if (fisheye1)
                {
                    double timeStamp = fisheye1.get_timestamp();
                    auto frameNumber = fisheye1.get_frame_number();
                    stringstream filename;

                    // cout << fisheye1.get_frame_timestamp_domain() << endl;

                    cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
                    filename << "../../Records/" << saveFolderName.str() << "/leftFisheye/left_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
                    cv::imwrite(filename.str() , img0);

                }

                if(fisheye2){ 

                    double timeStamp = fisheye2.get_timestamp();
                    auto frameNumber = fisheye2.get_frame_number();
                    stringstream filename;

                    cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);
                    filename << "../../Records/" << saveFolderName.str() << "/rightFisheye/right_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
                    cv::imwrite(filename.str() , img0);              
                }

                if(pose){

                    auto poseData = pose.get_pose_data();

                    double timeStamp = pose.get_timestamp();
                    t265_pose.write((const char*)(&timeStamp), sizeof(double));

                    writeVectorToFileBinary(t265_pose, poseData.translation);
                    writeVectorToFileBinary(t265_pose, poseData.velocity);
                    writeVectorToFileBinary(t265_pose, poseData.acceleration);
                    writeQuaternionToFileBinary(t265_pose, poseData.rotation);
                    writeVectorToFileBinary(t265_pose, poseData.angular_velocity);
                    writeVectorToFileBinary(t265_pose, poseData.angular_acceleration);

                    t265_pose.write((const char*)(&poseData.tracker_confidence), sizeof(unsigned int));
                    t265_pose.write((const char*)(&poseData.mapper_confidence), sizeof(unsigned int));


                }
            }

            if(shotCounter > numShots){
                continueRecord = false;
            }

        });
    }

    if(D435Connected) {
        
        pipe2.start(cfg2, [&](rs2::frame frame) {

            auto currentTime = chrono::steady_clock::now();
            auto passedTime = chrono::duration_cast<chrono::milliseconds>( currentTime - startTime ).count(); // in milliseconds

            if(passedTime > warmUpTime * 1000 && shotCounter <= numShots){ 

                if(!T265Connected && (passedTime - warmUpTime * 1000)/ (deltaT * 1000) > shotCounter){
                    cout << "shot taken" << endl;
                    shotCounter ++;
                }

                // Cast the frame to appropriate object type
                auto fset = frame.as<rs2::frameset>();
                auto motion = frame.as<rs2::motion_frame>();
                auto depth = fset.get_depth_frame();
                auto color = fset.get_color_frame();

                // If casting succeeded and the arrived frame is from gyro stream
                if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {

                    // Get gyro measures
                    rs2_vector gyro_data = motion.get_motion_data();

                    double timeStamp = motion.get_timestamp();
                    d435_gyro.write((const char*)(&timeStamp), sizeof(double));            
                    writeVectorToFileBinary(d435_gyro, gyro_data);

                }

                // If casting succeeded and the arrived frame is from accelerometer stream
                if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F) {

                    // Get accelerometer measures
                    rs2_vector accel_data = motion.get_motion_data();

                    double timeStamp = motion.get_timestamp();
                    d435_acc.write((const char*)(&timeStamp), sizeof(double));            
                    writeVectorToFileBinary(d435_acc, accel_data);

                }

                // If casting succeeded and the arrived frame is from depth stream
                if(depth) {
                    double timeStamp = depth.get_timestamp();
                    auto frameNumber = depth.get_frame_number();

                    // cout << "Depth frame num: " << frameNumber << endl;
                    stringstream filename;

                    cv::Mat img0(cv::Size(1280, 720), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
                    filename << "../../Records/" << saveFolderName.str() << "/depth/depth_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
                    cv::imwrite(filename.str() , img0);
                }

                // If casting succeeded and the arrived frame is from color stream
                if(color) {
                    double timeStamp = color.get_timestamp();
                    auto frameNumber = color.get_frame_number();
                    stringstream filename;

                    cv::Mat img0(cv::Size(1280, 720), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                    cv::Mat img1;
                    cv::cvtColor(img0, img1, cv::COLOR_BGR2RGB);
                    filename << "../../Records/" << saveFolderName.str() << "/rgb/rgb_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
                    cv::imwrite(filename.str() , img1);
                }

            }

            if(!T265Connected && shotCounter > numShots){
                continueRecord = false;
            }

        } );
        
    }

    while(continueRecord){
        usleep(10000);
    }

    usleep(1000000);
    if(T265Connected) pipe1.stop();
    if(D435Connected) pipe2.stop();

    //close the files

    t265_acc.close();
    t265_gyro.close();
    t265_pose.close();
    
    d435_acc.close();
    d435_gyro.close();


    
    return EXIT_SUCCESS;
}
catch (const error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}
