cmake_minimum_required(VERSION 3.0) 
project(mumax5core)

add_library(core STATIC 
    anisotropy.cu 
    anisotropy.hpp 
    cudaerror.hpp 
    datatypes.hpp 
    dynamicequation.hpp
    dynamicequation.cpp
    effectivefield.hpp 
    effectivefield.cu 
    exchange.hpp 
    exchange.cu 
    eulerstepper.cpp
    eulerstepper.hpp
    ferromagnet.cpp 
    ferromagnet.hpp 
    ferromagnetquantity.cpp 
    ferromagnetquantity.hpp 
    field.cu 
    field.hpp 
    fieldops.cu 
    fieldops.hpp 
    grid.cu 
    grid.hpp 
    quantity.cpp 
    quantity.hpp 
    stepper.cpp
    stepper.hpp
    system.cpp 
    system.hpp 
    timesolver.cpp
    timesolver.hpp
    torque.cu 
    torque.hpp 
    variable.hpp
    variable.cpp
    world.cpp 
    world.hpp
)

target_include_directories(core PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(core PUBLIC ${PROJECT_SOURCE_DIR})

set_target_properties(core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(core PROPERTIES POSITION_INDEPENDENT_CODE ON)