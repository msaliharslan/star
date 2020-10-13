
 
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
    std::string serial_number = "908412110459";
    auto device = get_device(serial_number);

    int i = 0;



    // open the profiles you need, or all of them
    auto sensor = device.first<rs2::sensor>();
    sensor.open(sensor.get_stream_profiles());

    double start = 0;



    // start the sensor providing a callback to get the frame
    sensor.start([&i, &start](rs2::frame f) {
       
        if (f.get_profile().stream_type() == RS2_STREAM_POSE) {
            double timeStamp = f.get_timestamp();
            if(start == 0)
                start = timeStamp;

            auto pose_frame = f.as<rs2::pose_frame>();
            cout << "hello amk" << endl;
            cout << "Pose_frame : " << scientific << f.get_timestamp() - start << endl;
        
        } else if (f.get_profile().stream_type() == RS2_STREAM_FISHEYE && f.get_profile().stream_index() == 1) {
            // this is the first fisheye imager

            double timeStamp = f.get_timestamp();
            if(start == 0)
                start = timeStamp;            

            auto fisheye_frame = f.as<rs2::video_frame>();

            stringstream filename;

            i++;

            cv::Mat img(cv::Size(848, 800), CV_8U, (void*)fisheye_frame.get_data(), cv::Mat::AUTO_STEP);
            filename.str("");
            filename << "maga_be" << i << ".png";
            //cv::imwrite(filename.str() , img);            

            cout << "Fisheye1 : " << scientific << f.get_timestamp() - start << endl;
        

        } else if (f.get_profile().stream_type() == RS2_STREAM_FISHEYE && f.get_profile().stream_index() == 2) {
            // this is the second fisheye imager
            auto fisheye_frame = f.as<rs2::video_frame>();

            double timeStamp = f.get_timestamp();
            if(start == 0)
                start = timeStamp;            
            
            cout << "Fisheye2 : " << scientific << f.get_timestamp() - start << endl;
                
        }
    });


    std::this_thread::sleep_for(std::chrono::microseconds(2000000));

    // and stop
    sensor.stop();
}