#include"wrappers.hpp"
#include"grid.hpp"

void wrap_grid(py::module& m) {

    py::class_<Grid>(m, "Grid", 
    "TODO: add Grid class documentation")

        .def(py::init<int3, int3>(),
              "Construct a grid with a given size and origin", 
              py::arg("size"), 
              py::arg("origin") = int3{0,0,0})

        .def_property("size", 
              &Grid::size, 
              &Grid::setSize, 
              "size of the grid")

        .def_property("origin", 
              &Grid::origin, 
              &Grid::setOrigin, 
              "origin of the grid");
}