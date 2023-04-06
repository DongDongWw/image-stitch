cmake_minimum_required(VERSION 3.1)
project(image_stitch)
set(CMAKE_BUILD_TYPE "Debug")

find_package(OpenCV REQUIRED)
include_directories("/usr/local/include/eigen3")


add_executable(stitch main.cpp  harris_self.cpp brief_self.cpp brute_force_match.cpp image_stitch.cpp ransac_homo.cpp rm_crack.cpp bright_consistency.cpp)
target_link_libraries(stitch ${OpenCV_LIBS})