#include "recorder_common.hpp"
#include "recorder_health_check.hpp"
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

using namespace std;
namespace fs = std::experimental::filesystem;

void initHealthCheck(int level) {
        //first generate the new recording directory name
    int currentSaveIndex = 0;

    //find currentSaveIndex
    std::string path = "../../Records/Health_Check/";
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
    for(int i = 0; i < level; i++) {
        string commandLevel = command1;
        commandLevel += "/level_" + to_string(i);
        system(commandLevel.c_str());
        
        string command2 = commandLevel;
        command2 += "/leftFisheye";
        system(command2.c_str());
        fisheye1_folder = path + "/leftFisheye";

        //create rightFisheye folder
        string command3 = commandLevel;
        command3 += "/rightFisheye";
        system(command3.c_str());
        fisheye2_folder = path + "/rightFisheye";

        //create rgb folder
        string command4 = commandLevel;
        command4 += "/rgb";
        system(command4.c_str());
        color_folder = path + "/rgb";

        //create depth folder
        string command5 = commandLevel;
        command5 += "/depth";
        system(command5.c_str()); 
        depth_folder = path + "/depth";

        //init file objects   
        string filesPath = p;
        filesPath += path;
        filesPath += saveFolderName.str();

        //init t265_acc.bin 
        t265_acc.open((filesPath + "/t265_acc.bin").c_str(), ios::out | ios::binary);

        //init d435_acc.bin
        d435_acc.open((filesPath + "/d435_acc.bin").c_str(), ios::out | ios::binary);

        //init t265_gyro.bin
        t265_gyro.open((filesPath + "/t265_gyro.bin").c_str(), ios::out | ios::binary);

        //init d435_acc.bin
        d435_gyro.open((filesPath + "/d435_gyro.bin").c_str(), ios::out | ios::binary);

        //pose
        t265_pose.open((filesPath + "/t265_pose.bin").c_str(), ios::out | ios::binary);    
    }
}