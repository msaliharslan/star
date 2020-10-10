#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <thread>
#include <fstream>
#include <experimental/filesystem>
#include <time.h>


using namespace rs2;
using namespace std;

namespace fs = std::experimental::filesystem;


int main(int argc, char **argv) try {


    //initialize

        //create folder

    //first generate the new recording directory name
    int currentSaveIndex = 0;

    //find currentSaveIndex
    std::string path = "../Records";
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


    fs::path p;
    p = fs::current_path();    
    string command = "mkdir ";
    command += p;    
    command += "/../Records/"; 
    command += saveFolderName.str();
    system(command.c_str());
        //init file objects

        //create folders

   
    rs2::config cfg1;
    rs2::device dev;
    rs2::colorizer color_map;
    rs2::context ctx;
    rs2::frameset frames;
    rs2::frame frame;

    // Obtain a list of devices currently present on the system

    auto devices = ctx.query_devices();
    size_t device_count = devices.size();
    if (!device_count){
        cout << "No device is found!!" << endl;
        return EXIT_SUCCESS;        
    }

  
    for (int i = 0; i < device_count; i++){
        cout << i + 1 << "\t" << devices[i].get_info(RS2_CAMERA_INFO_NAME) << endl;
    }


    cfg1.enable_device(devices[0].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg1.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg1.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    cfg1.enable_stream(RS2_STREAM_POSE);
    cfg1.enable_stream(RS2_STREAM_FISHEYE, 1);
    cfg1.enable_stream(RS2_STREAM_FISHEYE, 2);




    //T265

    pipeline pipe1(ctx);



    pipe1.start(cfg1, [&](rs2::frame frame)
    {
        // Cast the frame that arrived to motion frame
        auto fset = frame.as<rs2::frameset>();
        auto motion = frame.as<rs2::motion_frame>();
        auto fisheye1 = frame.as<rs2::frameset>().get_fisheye_frame(1);
        auto fisheye2 = frame.as<rs2::frameset>().get_fisheye_frame(2);

        // If casting succeeded and the arrived frame is from gyro stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get the timestamp of the current frame
            // ts2 = motion.get_timestamp();
            // Get gyro measures
            rs2_vector gyro_data = motion.get_motion_data();


        }
        // If casting succeeded and the arrived frame is from accelerometer stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get accelerometer measures
            rs2_vector accel_data = motion.get_motion_data();
        }

        if (fisheye1)
        {
            double timeStamp = fisheye1.get_timestamp();
            auto frameNumber = fisheye1.get_frame_number();
            stringstream filename;


            cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
            filename << "../Records/" << saveFolderName.str() << "/left_" << frameNumber << "_" << setw(10) << timeStamp << ".png";            
            cv::imwrite(filename.str() , img0);

        }

        if(fisheye2){


        }
    });


    int a;
    cin >> a;
    pipe1.stop();


    
    return EXIT_SUCCESS;
}
catch (const error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}
