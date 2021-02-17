#ifndef RECORDER_CALIBRATION
#define RECORDER_CALIBRATION

#include <librealsense2/rs.hpp>
#include <recorder_common.hpp>

const std::string calib_cases[] = {"flat_rot_0", 
                              "flat_rot_30",
                              "flat_rot_60",
                              "right_30_rot_0",
                              "right_30_rot_30",
                              "right_30_rot_60",
                              "right_60_rot_0",
                              "right_60_rot_30",
                              "right_60_rot_60",
                            //   "left_30_rot_0",
                            //   "left_30_rot_30",
                            //   "left_30_rot_60",
                            //   "left_60_rot_0",
                            //   "left_60_rot_30",
                            //   "left_60_rot_60",
                              "top_30_rot_0",
                              "top_30_rot_30",
                              "top_30_rot_60",
                              "top_60_rot_0",
                              "top_60_rot_30",
                              "top_60_rot_60",
                            //   "bottom_30_rot_0",
                            //   "bottom_30_rot_30",
                            //   "bottom_30_rot_60",
                            //   "bottom_60_rot_0",
                            //   "bottom_60_rot_30",
                            //   "bottom_60_rot_60", 
                              "top-right_45_rot_0",                     
                              "top-right_45_rot_30",                     
                              "top-right_45_rot_60"                     
                             };

extern std::string current_case;

void calibration_callback_d435(rs2::frame frame);
void calibration_callback_t265(rs2::frame frame);
void initCalibration();

#endif // RECORDER_CALIBRATION