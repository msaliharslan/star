#include "recorder_common.hpp"
#include "recorder_calibration.hpp"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

using namespace std;
namespace fs = std::experimental::filesystem;

string current_case = "";

void calibration_callback_d435(rs2::frame frame) {
    d435_pipe_mutex.lock();

    if(save_flag) {
        // Cast the frame to appropriate object type
        auto fset = frame.as<rs2::frameset>();
        auto depth = fset.get_depth_frame();
        auto color = fset.get_color_frame();

        // If casting succeeded and the arrived frame is from depth stream
        if(depth) {
            double timeStamp = depth.get_timestamp();
            unsigned long long frameNumber = getFrameNumber(depthFramePeriod, initialTimeStamp_d435, timeStamp);

            // cout << "Depth frame num: " << frameNumber << endl;
            stringstream filename;

            cv::Mat img0(cv::Size(width_depth, height_depth), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            filename << depth_folder + "/" + current_case +  "/depth/depth_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
            cv::imwrite(filename.str() , img0);
        }

        // If casting succeeded and the arrived frame is from color stream
        if(color) {
            double timeStamp = color.get_timestamp();
            unsigned long long frameNumber = getFrameNumber(depthFramePeriod, initialTimeStamp_d435, timeStamp);
            stringstream filename;

            cv::Mat img0(cv::Size(width_color, height_color), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
            cv::Mat img1;
            cv::cvtColor(img0, img1, cv::COLOR_BGR2RGB);
            filename << color_folder + "/" + current_case + "/rgb/rgb_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
            cv::imwrite(filename.str() , img1);
        }
    }
    else if(synch_flag) {
        auto depth = frame.as<rs2::frameset>().get_depth_frame();
        if(depth) {
            auto d435_frametimestamp = depth.get_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL);
            toas_d435[toas_d435_index] = d435_frametimestamp;
            toas_d435_index++;
        }
    }

    d435_pipe_mutex.unlock();
}

void calibration_callback_t265(rs2::frame frame) {
    t265_pipe_mutex.lock();

    if(save_flag) {
        // Cast the frame to appropriate object type
        auto fset = frame.as<rs2::frameset>();
        auto fisheye1 = fset.get_fisheye_frame(1);
        auto fisheye2 = fset.get_fisheye_frame(2);
        auto pose = frame.as<rs2::pose_frame>();

        if (fisheye1)
        {
            double timeStamp = fisheye1.get_timestamp();
            unsigned long long frameNumber = getFrameNumber(fisheyeFramePeriod, initialTimeStamp_t265, timeStamp);
            stringstream filename;

            cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye1.get_data(), cv::Mat::AUTO_STEP);
            filename << fisheye1_folder + "/" + current_case + "/leftFisheye/left_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";     
            cv::imwrite(filename.str() , img0);

        }

        if(fisheye2){ 

            double timeStamp = fisheye2.get_timestamp();
            unsigned long long frameNumber = getFrameNumber(fisheyeFramePeriod, initialTimeStamp_t265, timeStamp);
            stringstream filename;

            cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);
            filename << fisheye2_folder + "/" + current_case + "/rightFisheye/right_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
            cv::imwrite(filename.str() , img0);              
        }
    }

    else if(synch_flag) {
        auto fisheye = frame.as<rs2::frameset>().get_fisheye_frame();
        if(fisheye) {
            auto t265_frametimestamp = fisheye.get_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL);
            toas_t265[toas_t265_index] = t265_frametimestamp;
            toas_t265_index++;
        }
    }

    t265_pipe_mutex.unlock();
}

void initCalibration() {
    //first generate the new recording directory name
    int currentSaveIndex = 0;

    //find currentSaveIndex
    std::string path = "../../Records/Calibration/";
    for (const auto & entry : fs::directory_iterator(path)){
        std::string delimiter1 = "/";
        std::string delimeter2 = "_";
        std::string entryPath = std::string(entry.path());
        std::reverse(entryPath.begin(), entryPath.end());
        std::string fileName = (entryPath.substr(0, entryPath.find(delimiter1)));
        std::reverse(fileName.begin(), fileName.end());
        std::string saveIndexString = fileName.substr(0, fileName.find(delimeter2));
        int index = stoi(saveIndexString);
        if(++index > currentSaveIndex)
            currentSaveIndex = index;        
    }
    time_t rawtime;
    struct tm * timeinfo;

    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime (buffer,80,"_%F_%R\0AA",timeinfo);

    std::stringstream saveFolderName;
    saveFolderName.str("");
    saveFolderName << "/" << currentSaveIndex;
    saveFolderName << buffer;

    

    //create main folder
    fs::path p;
    p = fs::current_path();    
    string command1 = "mkdir ";
    command1 += p;    
    command1 += "/" + path; 
    command1 += saveFolderName.str();
    system(command1.c_str());

    for (const string & cas : calib_cases){
        string commandCase = command1;
        commandCase += "/" +  cas ;
        system(commandCase.c_str());

        //create leftFisheye folder
        string command2 = command1;
        command2 += "/" + cas + "/leftFisheye";
        system(command2.c_str());
        fisheye1_folder = path + saveFolderName.str();

        //create rightFisheye folder
        string command3 = command1;
        command3 += "/" + cas + "/rightFisheye";
        system(command3.c_str());
        fisheye2_folder = path + saveFolderName.str();

        //create rgb folder
        string command4 = command1;
        command4 += "/" + cas + "/rgb";
        system(command4.c_str());
        color_folder = path + saveFolderName.str();

        //create depth folder
        string command5 = command1;
        command5 += "/" + cas + "/depth";
        system(command5.c_str()); 
        depth_folder = path + saveFolderName.str();

    }
}