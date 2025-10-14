cmake_minimum_required(VERSION 3.24)
option(AUTO_DETECT_CUDA_ARCH "Detect GPU archs at configure time" OFF)
if(NOT AUTO_DETECT_CUDA_ARCH)
  # Real SASS + PTX (virtual) für die höchste CC
  set(CMAKE_CUDA_ARCHITECTURES
      "60;61;62;70;72;75;80;80-virtual"
      CACHE STRING "CUDA SM list" FORCE)
endif()
project(mumaxplus LANGUAGES CUDA CXX)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS TRUE)
endif()

#set(CMAKE_CUDA_ARCHITECTURES "60;61;62;70;72;75;80" CACHE STRING "CUDA SM list" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -Wall -Xptxas -O3" CACHE STRING "" FORCE)

add_definitions(-DFP_PRECISION=SINGLE) # FP_PRECISION should be SINGLE or DOUBLE
add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

add_subdirectory(src/bindings)
add_subdirectory(src/cmd)
add_subdirectory(src/core)
add_subdirectory(src/cudautil)
add_subdirectory(src/linsolver)
add_subdirectory(src/physics)
