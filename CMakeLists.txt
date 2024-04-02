cmake_minimum_required(VERSION 3.16)
project(mumax5 LANGUAGES CUDA CXX)

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

# automatically detect best CUDA architecture
# this is bad undocumented code which works
# https://stackoverflow.com/a/68223399
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

add_definitions(-DFP_PRECISION=SINGLE) # FP_PRECISION should be SINGLE or DOUBLE
add_definitions(-D_USE_MATH_DEFINES) # Needed for cmath constants on Windows

add_subdirectory(src/bindings)
add_subdirectory(src/cmd)
add_subdirectory(src/core)
add_subdirectory(src/cudautil)
add_subdirectory(src/linsolver)
add_subdirectory(src/physics)
