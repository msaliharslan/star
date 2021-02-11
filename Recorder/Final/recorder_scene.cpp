#include "recorder_scene.hpp"
#include "recorder_common.hpp"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

using namespace std;
namespace fs = std::experimental::filesystem;

void scene_callback_d435(rs2::frame frame) {
    d435_pipe_mutex.lock();

    if(save_flag) {
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

            cout << "Depth frame num: " << frameNumber << endl;
            stringstream filename;

            cv::Mat img0(cv::Size(1280, 720), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
            filename << depth_folder + "/depth_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
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
            filename << color_folder + "/rgb_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
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


void scene_callback_t265(rs2::frame frame) {

    t265_pipe_mutex.lock();

    if(save_flag) {
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
            filename << fisheye1_folder + "/left_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";     
            cv::imwrite(filename.str() , img0);

        }

        if(fisheye2){ 

            double timeStamp = fisheye2.get_timestamp();
            auto frameNumber = fisheye2.get_frame_number();
            stringstream filename;

            cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)fisheye2.get_data(), cv::Mat::AUTO_STEP);
            filename << fisheye2_folder + "/right_" << frameNumber << "_" << std::setprecision(13) << timeStamp << ".png";            
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

void initScene(string name) {
    //first generate the new recording directory name
    int currentSaveIndex = 0;

    //find currentSaveIndex
    std::string path = "../../Records/Scene/" + name;
    std::stringstream scene_name_cmd;
    scene_name_cmd.str("");
    scene_name_cmd << "if [ ! -d ";
    scene_name_cmd << path << " ]; then mkdir " << path << "; fi";
    system(scene_name_cmd.str().c_str());
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
    //create leftFisheye folder
    string command2 = command1;
    command2 += "/leftFisheye";
    system(command2.c_str());
    fisheye1_folder = path + saveFolderName.str() + "/leftFisheye";

    //create rightFisheye folder
    string command3 = command1;
    command3 += "/rightFisheye";
    system(command3.c_str());
    fisheye2_folder = path + saveFolderName.str() + "/rightFisheye";

    //create rgb folder
    string command4 = command1;
    command4 += "/rgb";
    system(command4.c_str());
    color_folder = path + saveFolderName.str() +  "/rgb";

    //create depth folder
    string command5 = command1;
    command5 += "/depth";
    system(command5.c_str()); 
    depth_folder = path + saveFolderName.str() + "/depth";

    //init file objects   
    string filesPath = p;
    filesPath += "/" + path;
    filesPath += saveFolderName.str();

    //init t265_acc.bin 
    t265_acc.open((filesPath + "/t265_acc.bin").c_str(), ios::out | ios::binary);
    cout << (filesPath + "/t265_acc.bin").c_str() << endl;

    //init d435_acc.bin
    d435_acc.open((filesPath + "/d435_acc.bin").c_str(), ios::out | ios::binary);

    //init t265_gyro.bin
    t265_gyro.open((filesPath + "/t265_gyro.bin").c_str(), ios::out | ios::binary);

    //init d435_acc.bin
    d435_gyro.open((filesPath + "/d435_gyro.bin").c_str(), ios::out | ios::binary);

    //pose
    t265_pose.open((filesPath + "/t265_pose.bin").c_str(), ios::out | ios::binary);    

}