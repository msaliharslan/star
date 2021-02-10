#include <librealsense2/rs.hpp> 
#include <iostream>
#include "recorder_calibration.hpp"
#include "recorder_health_check.hpp"
#include "recorder_scene.hpp"
#include "recorder_common.hpp"
#include <chrono>
#include <thread>
#include <mutex>
#include <experimental/random>

using namespace rs2;
using namespace std;

bool timeSyncher(pipeline *pipe1, config cfg1, void (*callback1)(frame), pipeline *pipe2, config cfg2, void (*callback2)(frame), bool save) {

    while(true) {
        rs2::context ctx;
        bool D435Connected = false;
        bool T265Connected = false;
        auto devices = ctx.query_devices();
        size_t device_count = devices.size();
        if (!device_count){
            cout << "No device is found!!" << endl;
            exit(2);        
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
                cfg2.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
                cfg2.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
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

        pipe1 = new pipeline(ctx);
        pipe2 = new pipeline(ctx);

        if(T265Connected && D435Connected) {
            pipe1->start(cfg1, callback1);
            pipe2->start(cfg2, callback2);
        }
        else {
            cerr << "Please connect both D435i and T265 cameras!!!" << endl;
            exit(1);
        }

        for(int n = 0; n < 5; n++) {

            std::this_thread::sleep_for(std::chrono::milliseconds(500) );
            d435_pipe_mutex.lock();
            t265_pipe_mutex.lock();


            for(int i=0; i < toas_d435_index; i++){
                for(int j=0; j < toas_t265_index; j++){
                    if( abs(toas_d435[i] - toas_t265[j]) < time_difference_threshold ){
                        cout << "Cameras are now sycnhed!!!" << endl;
                        synch_flag = false;
                        d435_pipe_mutex.unlock();
                        t265_pipe_mutex.unlock();
                        return true;
                    } 
                }
            }
            d435_pipe_mutex.unlock();
            t265_pipe_mutex.unlock();
        }

        delete pipe1; pipe1 = nullptr;
        delete pipe2; pipe2 = nullptr;

        std::this_thread::sleep_for(std::chrono::milliseconds(std::experimental::randint(0, 1000)) );
    }

}