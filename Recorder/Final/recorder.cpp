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
#include <iterator>
#include <boost/program_options.hpp>
#include <exception>
#include "recorder_calibration.hpp"
#include "recorder_health_check.hpp"
#include "recorder_scene.hpp"
#include "recorder_common.hpp"
#include "time_syncher.hpp"

using namespace rs2;
using namespace std;

namespace fs = std::experimental::filesystem;
namespace po = boost::program_options;


int main(int argc, char **argv) try {

    int type, duration, health_check_level;
    string scene_name;
    bool synched;
    context ctx;
    char input_cmd = '\0';

    pipeline *pipe1;
    pipeline *pipe2;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message") 
    ("type", po::value<int>(), "set type")
    ("name", po::value<string>(), "set name")
    ("duration", po::value<int>(), "set duration")
    ("level", po::value<int>(), "set health check level");


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("type")) {
        type = vm["type"].as<int>();
    }
    else {
        cout << "Must enter type code!!" << endl;
        return 0;
    }

    switch(type) {
        case 0: // scene
            if(vm.count("name")) {
                scene_name = vm["name"].as<string>();
            }
            else {
                cout << "Must enter scene name for scene recording!!" << endl;
                return 4;
            }

            if(vm.count("duration")) {
                duration = vm["duration"].as<int>();
            }
            else {
                cout << "Must enter duration for scene recording!!" << endl;
                return 5;
            }
            initScene(scene_name);
            synched = timeSyncher(pipe1, scene_callback_d435, pipe2, scene_callback_t265);
            cout << "Press s to start recording" << endl;
            while(input_cmd != 's'){
                cin >> input_cmd;
            }
            save_flag = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(duration * 1000) );
            save_flag = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(100) );
            break;

        case 1: // health check
            if(vm.count("level")) {
                health_check_level = vm["level"].as<int>();
            }
            else {
                cout << "Must enter health check level for health check recording!!" << endl;
                return 0;
            }
            initHealthCheck(health_check_level);
            synched = timeSyncher(pipe1, health_check_callback_d435, pipe2, health_check_callback_t265);
            for(level_index = 0; level_index < health_check_level; level_index++) {
                cout << "Press r when you are ready" << endl;
                while(input_cmd != 'r'){
                    cin >> input_cmd;
                }
                input_cmd = '\0';
                save_flag = true;
                std::this_thread::sleep_for(std::chrono::milliseconds(3000) );
                save_flag = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100) );
            }
            break;

        case 2: // calibration

            initCalibration();
            synched = timeSyncher(pipe1, calibration_callback_d435, pipe2, calibration_callback_t265);
            for (const string & cas : calib_cases){
                current_case = cas; 
                cout << "Positon board as " << cas << endl;
                cout << "Press r when you are ready" << endl;
                while(input_cmd != 'r'){
                    cin >> input_cmd;
                }
                input_cmd = '\0';
                save_flag = true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2000) );
                save_flag = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100) );
            }           
            break;

        default:
            cout << "Unknown type code!!" << endl;
            return 6;
    }

    pipe1->stop();
    pipe2->stop();
    cout << "pipes are stopped" << endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(100) );
    delete pipe1; pipe1 = nullptr;
    delete pipe2; pipe2 = nullptr;
    cout << "pipes are deleted" << endl;
    //close the files
    if(type == 0){
        t265_acc.close();
        t265_gyro.close();
        t265_pose.close();
        d435_acc.close();
        d435_gyro.close();
    }
    cout << "files are closed" << endl;
    return 0;
}

catch (const error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}

