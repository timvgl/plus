add_subdirectory(pybind11)

pybind11_add_module(engine
    cast.hpp
    wrappers.hpp
    ferromagnet.cpp
    field.cpp
    grid.cpp
    main.cpp
    quantity.cpp
    timesolver.cpp
    variable.cpp
    world.cpp
)

target_include_directories(engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(engine PRIVATE physics)