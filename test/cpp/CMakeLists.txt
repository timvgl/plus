project(mumax5)

add_executable(sandbox sandbox.cpp)
target_link_libraries(sandbox PRIVATE physics)
target_include_directories(sandbox PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})