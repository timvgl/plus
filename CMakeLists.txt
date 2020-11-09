cmake_minimum_required(VERSION 3.10)
project(mumax5 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

add_definitions(-DFP_PRECISION=SINGLE) # FP_PRECISION should be SINGLE or DOUBLE
add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

add_subdirectory(src/core)
add_subdirectory(src/bindings)
add_subdirectory(src/physics)

add_subdirectory(test/cpp)
