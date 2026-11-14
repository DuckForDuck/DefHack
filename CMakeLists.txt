cmake_minimum_required(VERSION 3.16)
project(thermal_isp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

add_executable(thermal_isp
  src/main.cpp
  src/stages.cpp
  src/utils.cpp
  src/video.cpp)
target_include_directories(thermal_isp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(thermal_isp PRIVATE ${OpenCV_LIBS})
if(NOT MSVC)
  target_compile_options(thermal_isp PRIVATE -O3 -Wall -Wextra)
endif()
