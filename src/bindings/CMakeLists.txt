project(mumax5core)

add_subdirectory(pybind11)

pybind11_add_module(mumax5
    cast.hpp
    wrappers.hpp
    ferromagnet.cpp
    field.cpp
    grid.cpp
    main.cpp
    quantity.cpp
    variable.cpp
    world.cpp
)

target_include_directories(mumax5 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mumax5 PRIVATE core)