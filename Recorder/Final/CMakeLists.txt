cmake_minimum_required(VERSION 3.1.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
SET(PLATFORM_SPECIFIC_LIBS "-lpthread")



project(record)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.65.1 EXACT COMPONENTS program_options REQUIRED) # due to some error in salihs pc

add_executable(record recorder.cpp recorder_calibration.cpp recorder_common.cpp recorder_health_check.cpp recorder_scene.cpp time_syncher.cpp)
include_directories("/hdd/realsense/librealsense")

include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(record PUBLIC ${Boost_LIBRARIES})
set_property(TARGET record PROPERTY CXX_STANDARD 17)
target_link_libraries(record PUBLIC ${REALSENSE2_FOUND} stdc++fs )
target_link_libraries(record PRIVATE ${OpenCV_LIBS})

include_directories(${CMAKE_SOURCE_DIR})

