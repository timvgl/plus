add_subdirectory(pybind11)

pybind11_add_module(engine
    cast.hpp
    main.cpp
    wrappers.hpp
    wrap_debug.cpp
    wrap_ferromagnet.cpp
    wrap_field.cpp
    wrap_grid.cpp
    wrap_parameter.cpp
    wrap_fieldquantity.cpp
    wrap_timesolver.cpp
    wrap_variable.cpp
    wrap_world.cpp
)

target_include_directories(engine PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(engine PRIVATE physics)