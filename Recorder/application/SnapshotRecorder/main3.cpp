#include <librealsense2/rs.hpp> 
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <ctime>    

using namespace std;

int main(int argc, char **argv){

    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::colorizer color_map;
    rs2::context ctx;
    rs2::frameset frames;
    rs2::frame frame;



    auto devices = ctx.query_devices();
    int device_count = devices.size();



    if (!device_count){
        cout << "No device is found!!" << endl;
        return EXIT_SUCCESS;        
    }

    if (device_count == 1){

        auto dev1 = devices[0];
        cout << "Only one device is found: " << dev1.get_info(RS2_CAMERA_INFO_NAME) << " configuring" << endl;
        cfg.enable_device(dev1.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        cfg.enable_all_streams();
        
        
        //cfg.enable_stream(RS2_STREAM_FISHEYE, 1, RS2_FORMAT_ANY, 0);
        //cfg.enable_stream(RS2_STREAM_FISHEYE, 2, RS2_FORMAT_ANY, 0);

        //cfg.enable_record_to_file("record.bag");
    }

    else {      
        for (int i = 0; i < device_count; i++){
            // cout << i + 1 << "\t" << devices[i].get_info(RS2_CAMERA_INFO_NAME) << endl;
            // not implemented yet
        }
    }
    

    stringstream filename;
    cout << "hello priorrr" << endl;
    pipe.start(cfg);
    cout << "hellooo " << endl;

    int numberOfLeft = 0;
    int numberOfRight = 0;

    double diff1 = 0;
    double diff2 = 0;

    double temp1 = 0;
    double temp2 = 0;

    for(int i = 0; i < 10; i++){

        frames = pipe.wait_for_frames();
        auto frame0 = frames.get_fisheye_frame(1);
        auto frame1 = frames.get_fisheye_frame(2);

        // cout << frame0.get_profile().stream_index() << endl;
        // cout << frame1.get_profile().stream_index() << endl;

        if(i > 0){
            diff1 = frame0.get_timestamp() - temp1;
            diff2 = frame1.get_timestamp() - temp2;
        }

        temp1 = frame0.get_timestamp();
        temp2 = frame1.get_timestamp();


        cv::Mat img0(cv::Size(848, 800), CV_8U, (void*)frame0.get_data(), cv::Mat::AUTO_STEP);
        filename.str("");
        filename << i << "_0.png";
        //cv::imwrite(filename.str() , img0);

        cv::Mat img1(cv::Size(848, 800), CV_8U, (void*)frame1.get_data(), cv::Mat::AUTO_STEP);
        filename.str("");
        filename << i << "_1.png";
        //cv::imwrite(filename.str() , img1);


    }

    
           
    // cv::namedWindow("record", cv::WINDOW_AUTOSIZE);
    // cv::imshow("record", img);
    // cv::waitKey();
    return EXIT_SUCCESS;
}