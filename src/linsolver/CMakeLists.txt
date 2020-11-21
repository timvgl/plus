cmake_minimum_required(VERSION 3.18) 
project(mumax5linsolver)

add_library(linsolver STATIC 
    linsolver.cpp
    linsolver.hpp
    linsystem.cu
    linsystem.hpp
    vec.cu
    vec.hpp
)

target_compile_definitions(linsolver PUBLIC -DLS_FP_PRECISION=DOUBLE) # should be SINGLE or DOUBLE 
target_include_directories(linsolver PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(linsolver PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(linsolver PUBLIC core)

set_target_properties(linsolver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(linsolver PROPERTIES POSITION_INDEPENDENT_CODE ON)