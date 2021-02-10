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

using namespace rs2;
using namespace std;

namespace fs = std::experimental::filesystem;
namespace po = boost::program_options;


int main(int argc, char **argv) try {

    int type, duration, health_check_level;
    string scene_name;

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
                cout << scene_name << endl;
            }
            else {
                cout << "Must enter scene name for scene recording!!" << endl;
                return 0;
            }

            if(vm.count("duration")) {
                duration = vm["duration"].as<int>();
            }
            else {
                cout << "Must enter duration for scene recording!!" << endl;
                return 0;
            }
            initScene(scene_name);
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
            break;

        case 2: // calibration

            initCalibration();
            break;

        default:
            cout << "Unknown type code!!" << endl;
            return 0;
    }
}

catch (const error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}

