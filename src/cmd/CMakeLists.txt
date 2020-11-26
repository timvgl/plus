add_executable(cmd main.cpp)

target_include_directories(cmd PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(cmd PUBLIC physics)
target_link_libraries(cmd PUBLIC core)
