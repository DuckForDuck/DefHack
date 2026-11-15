cmake_minimum_required(VERSION 3.16)
project(thermal_isp C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)  # gnu11: vector extensions, POSIX calls
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
option(PORTABLE "build without -march/-mcpu=native" OFF)

find_package(Threads REQUIRED)

add_executable(thermal_isp
  src/main.c
  src/stages.c
  src/dis.c
  src/filters.c
  src/parallel.c
  src/video.c)
target_link_libraries(thermal_isp PRIVATE Threads::Threads m)
# -O3 turns on the vectoriser; -fno-math-errno / -fno-trapping-math let it
# vectorise sqrt and divisions. No -ffast-math: results stay IEEE-exact.
target_compile_options(thermal_isp PRIVATE -O3 -fno-math-errno -fno-trapping-math
  -Wall -Wextra -Wno-unused-parameter -Wno-psabi)

if(NOT PORTABLE)
  include(CheckCCompilerFlag)
  check_c_compiler_flag(-mcpu=native HAVE_MCPU_NATIVE)
  check_c_compiler_flag(-march=native HAVE_MARCH_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" AND HAVE_MCPU_NATIVE)
    target_compile_options(thermal_isp PRIVATE -mcpu=native)
  elseif(HAVE_MARCH_NATIVE)
    target_compile_options(thermal_isp PRIVATE -march=native)
  endif()
endif()
