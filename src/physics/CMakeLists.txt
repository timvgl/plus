cmake_minimum_required(VERSION 3.0) 
project(mumax5physics)

add_library(physics STATIC 
    anisotropy.cu 
    anisotropy.hpp 
    constants.hpp
    demag.cu
    demag.hpp
    demagkernel.cu
    demagkernel.hpp
    demagconvolution.cu
    demagconvolution.hpp
    effectivefield.hpp 
    effectivefield.cpp
    energy.hpp 
    energy.cpp
    exchange.hpp 
    exchange.cu 
    ferromagnet.cpp 
    ferromagnet.hpp 
    ferromagnetquantity.cpp 
    ferromagnetquantity.hpp 
    minimizer.cu
    minimizer.hpp
    newell.cu
    newell.hpp
    system.cpp 
    system.hpp 
    torque.cu 
    torque.hpp 
    world.cpp 
    world.hpp
    zeeman.cu
    zeeman.hpp
)

target_include_directories(physics PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(physics PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(physics PUBLIC core)

link_directories(${CUDA_LIBRARY_DIR})
target_link_libraries(physics PRIVATE cufft)


set_target_properties(physics PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(physics PROPERTIES POSITION_INDEPENDENT_CODE ON)