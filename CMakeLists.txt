cmake_minimum_required(VERSION 3.14)
project(bv_reconstruction)

set(CMAKE_CXX_STANDARD 14)

add_executable(bv_reconstruction reconstruction.cpp reconstruction.h other.h other.cpp main.cpp WienerFilter.cpp WienerFilter.h SGSmooth.cpp SGSmooth.hpp)
find_package(OpenCV REQUIRED)
target_link_libraries(bv_reconstruction ${OpenCV_LIBS})
