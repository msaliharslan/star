#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <string>

using namespace rs2;
using namespace std;

// void collectAccData(device dev){
//     config cfg;
//     cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_ASIC_SERIAL_NUMBER));
//     cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
//     cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
//     cfg.enable_record_to_file(string(dev.get_info(RS2_CAMERA_INFO_NAME)) + ".bag");

//     pipeline pipe;
//     auto profile = pipe.start(cfg, [&](rs2::frame frame)
//     {
//         // Cast the frame that arrived to motion frame
//         auto motion = frame.as<rs2::motion_frame>();
//         // If casting succeeded and the arrived frame is from gyro stream
//         if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
//         {
//             // Get the timestamp of the current frame
//             double ts = motion.get_timestamp();
//             // Get gyro measures
//             rs2_vector gyro_data = motion.get_motion_data();
//             // Call function that computes the angle of motion based on the retrieved measures
//             algo.process_gyro(gyro_data, ts);
//         }
//         // If casting succeeded and the arrived frame is from accelerometer stream
//         if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
//         {
//             // Get accelerometer measures
//             rs2_vector accel_data = motion.get_motion_data();
//             // Call function that computes the angle of motion based on the retrieved measures
//             algo.process_accel(accel_data);
//         }
//     });

// }

int main(int argc, char **argv) try {

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
    cfg2.enable_stream(RS2_STREAM_POSE);
    cfg2.enable_record_to_file(string(devices[1].get_info(RS2_CAMERA_INFO_NAME)) + ".bag");

    pipeline pipe1(ctx), pipe2(ctx);
    pipe1.start(cfg1);
    pipe2.start(cfg2);

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
