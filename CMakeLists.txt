cmake_minimum_required(VERSION 3.10)
project(mumax5 LANGUAGES CUDA CXX)

add_subdirectory(src/core)
add_subdirectory(src/bindings)
add_subdirectory(test/cpp)
