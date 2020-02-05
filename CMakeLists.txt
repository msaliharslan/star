cmake_minimum_required(VERSION 3.1.0)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

project(record)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(record main.cpp)
include_directories("/hdd/realsense/librealsense")

set_property(TARGET record PROPERTY CXX_STANDARD 11)
target_link_libraries(record PUBLIC ${REALSENSE2_FOUND} )
target_link_libraries(record PRIVATE ${OpenCV_LIBS})


include_directories(record)

