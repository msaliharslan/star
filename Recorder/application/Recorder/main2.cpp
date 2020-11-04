
 
#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <string>


using namespace std;rs2::device get_device(const std::string& serial_number) {
    rs2::context ctx;
    while (true) {
        for (auto&& dev : ctx.query_devices())
            if (std::string(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)) == serial_number)
                return dev;
        

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main() {



    // get the device given the serial number
    std::string serial_number = "850312070185"; // fisheye = "908412110459";
    auto device = get_device(serial_number);

    cout << device.get_info(RS2_CAMERA_INFO_NAME) << endl;; 
    int i = 0;



    // open the profiles you need, or all of them
    auto sensor = device.first<rs2::sensor>();
    sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 0);
    for(auto a : sensor.get_stream_profiles() ){
        // cout << a.stream_name() << a.unique_id() << endl;
        if( a.unique_id() == 0){
            sensor.open(a);
            break;
        }
    }
    // sensor.open(sensor.get_stream_profiles()[0]);
    cout << "Sensor is opened!!" << endl;
    double start = 0;



    // // start the sensor providing a callback to get the frame
    // sensor.start([&i, &start](rs2::frame f) {
       
    //     if (f.get_profile().stream_type() == RS2_STREAM_POSE) {
    //         double timeStamp = f.get_timestamp();
    //         if(start == 0)
    //             start = timeStamp;

    //         auto pose_frame = f.as<rs2::pose_frame>();
    //         cout << "hello amk" << endl;
    //         cout << "Pose_frame : " << scientific << f.get_timestamp() - start << endl;
        
    //     } else if (f.get_profile().stream_type() == RS2_STREAM_FISHEYE && f.get_profile().stream_index() == 1) {
    //         // this is the first fisheye imager

    //         double timeStamp = f.get_timestamp();
    //         if(start == 0)
    //             start = timeStamp;            

    //         auto fisheye_frame = f.as<rs2::video_frame>();

    //         stringstream filename;

    //         i++;

    //         cv::Mat img(cv::Size(848, 800), CV_8U, (void*)fisheye_frame.get_data(), cv::Mat::AUTO_STEP);
    //         filename.str("");
    //         filename << "maga_be" << i << ".png";
    //         //cv::imwrite(filename.str() , img);            

    //         cout << "Fisheye1 : " << scientific << f.get_timestamp() - start << endl;
        

    //     } else if (f.get_profile().stream_type() == RS2_STREAM_FISHEYE && f.get_profile().stream_index() == 2) {
    //         // this is the second fisheye imager
    //         auto fisheye_frame = f.as<rs2::video_frame>();

    //         double timeStamp = f.get_timestamp();
    //         if(start == 0)
    //             start = timeStamp;            
            
    //         cout << "Fisheye2 : " << scientific << f.get_timestamp() - start << endl;
                
    //     }
    // });

    // For D435i
    sensor.start([&i, &start](rs2::frame f) {
       
        if (f.get_profile().stream_type() == RS2_STREAM_DEPTH) {

            double timeStamp = f.get_timestamp();
            if(start == 0)
                start = timeStamp;            

            auto depth_frame = f.as<rs2::video_frame>();

            stringstream filename;

            i++;

            cv::Mat img(cv::Size(1280, 720), CV_16U, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
            filename.str("");
            filename << i << "_" << timeStamp << ".png";
            cv::imwrite(filename.str() , img);            

            cout << "Depth : " << scientific << timeStamp << endl;
            cout << "Depth frame number :" << f.get_frame_number() << endl;
        

        } else if(f.get_profile().stream_type() == RS2_STREAM_COLOR ) {
            cout << "color frame arrived" << endl;
        }
    });


    std::this_thread::sleep_for(std::chrono::microseconds(2000000));

    // and stop
    sensor.stop();
}