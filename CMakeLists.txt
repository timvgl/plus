cmake_minimum_required(VERSION 3.10)
project(mumax5 LANGUAGES CUDA CXX)

set (CMAKE_CXX_STANDARD 17)

add_subdirectory(src/core)
add_subdirectory(src/bindings)
add_subdirectory(src/physics)

add_subdirectory(test/cpp)
