cmake_minimum_required(VERSION 3.1)
project(image_stitch)
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
include_directories("/usr/local/include/eigen3")


add_executable(stitch main.cpp harris_opencv.cpp harris_self.cpp brief_self.cpp brute_force_match.cpp brief_gao.cpp image_wrap.cpp ransac_homo.cpp)
target_link_libraries(stitch ${OpenCV_LIBS})

add_executable(orb_opencv orb_opencv.cpp)
target_link_libraries(orb_opencv ${OpenCV_LIBS})