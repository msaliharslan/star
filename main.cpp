#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>
#include <thread>

using namespace rs2;
using namespace std;

const size_t len = 50;
const size_t frame_number_wait = 30;
long long toa1[len], toa2[len];
int index1 = 0, index2 = 0, append_flag = 0;

void recordValidator(){
    const int time_difference_threshold = 10;
    while(true){
        if(index1 == index2 && index1 == len){

            for(int i=0; i < len; i++){
                for(int j=0; j < len; j++){
                    if( abs(toa1[i] - toa2[j]) < time_difference_threshold ){
                        cout << "\tvay anam vay babam" << endl;
                        return;
                    } 
                }
            }
            cout << "\tzaaaaaaaaaaa" << endl;
            return;
        }
        sleep(1);
    }
}


int main(int argc, char **argv) try {

    std::thread Flag (recordValidator);
   
    rs2::config cfg1, cfg2;
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
    cfg1.enable_stream(RS2_STREAM_DEPTH);
    cfg1.enable_record_to_file(string(devices[0].get_info(RS2_CAMERA_INFO_NAME)) + ".bag");

    cfg2.enable_device(devices[1].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg2.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg2.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    // cfg2.enable_stream(RS2_STREAM_POSE);
    cfg2.enable_stream(RS2_STREAM_FISHEYE, 1);
    cfg2.enable_stream(RS2_STREAM_FISHEYE, 2);
    cfg2.enable_record_to_file(string(devices[1].get_info(RS2_CAMERA_INFO_NAME)) + ".bag");


    double ts1, ts2;
    pipeline pipe1(ctx), pipe2(ctx);
    pipe1.start(cfg1, [&](rs2::frame frame)
    {
        // Cast the frame that arrived to motion frame
        auto fset = frame.as<rs2::frameset>();
        auto depth = fset.get_depth_frame();
        auto motion = frame.as<rs2::motion_frame>();
        // If casting succeeded and the arrived frame is from gyro stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get the timestamp of the current frame
            // ts1 = motion.get_timestamp();
            // Get gyro measures
            rs2_vector gyro_data = motion.get_motion_data();
        }
        // If casting succeeded and the arrived frame is from accelerometer stream
        if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
        {
            // Get accelerometer measures
            rs2_vector accel_data = motion.get_motion_data();
        }
        // If casting succeeded and ..
        if (fset)
        {
            auto depthMetaData_framenumber = depth.get_frame_number();
            // cout << "Frame Number of Depth: " << depthMetaData_framenumber << endl;
            if (depthMetaData_framenumber > frame_number_wait)
            {
                if (depthMetaData_framenumber == frame_number_wait + 1) append_flag ++;
                // ts1 = depth.get_timestamp();
                // cout << "depth: " << ts1 << endl;
                auto depthMetaData_frametimestamp = depth.get_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL);
                // cout << "Frame Timestamp : " << depthMetaData_frametimestamp << endl;
                if(append_flag == 2 && index1 < len){
                    toa1[index1] = depthMetaData_frametimestamp;
                    index1++;
                }
            }
        }
    });
    pipe2.start(cfg2, [&](rs2::frame frame)
    {
        // Cast the frame that arrived to motion frame
        auto fset = frame.as<rs2::frameset>();
        auto motion = frame.as<rs2::motion_frame>();
        auto fisheye = frame.as<rs2::frameset>().get_fisheye_frame();
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

        if (fisheye)
        {
            auto fisheyeMetaData_framenumber = fisheye.get_frame_number();
            // cout << "Frame Number of fisheye: " << fisheyeMetaData_framenumber << endl;
            if (fisheyeMetaData_framenumber > frame_number_wait)
            {
                if (fisheyeMetaData_framenumber == frame_number_wait + 1) append_flag ++;
                ts2 = fisheye.get_timestamp();
                // cout << "fisheye: " << (long long)(ts2) << endl;
                // auto fisheyeMetaData_frametimestamp = fisheye.get_frame_metadata(RS2_FRAME_METADATA_TIME_OF_ARRIVAL);
                // cout << "Frame Timestamp(f) : " << fisheyeMetaData_frametimestamp << endl;
                if(append_flag == 2 && index1 < len){
                    toa2[index2] = ts2;
                    index2++;
                }
            }
        }

    });

    int a;
    cin >> a;
    pipe1.stop();
    pipe2.stop();


    
    return EXIT_SUCCESS;
}
catch (const error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}
