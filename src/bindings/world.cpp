#include"wrappers.hpp"
#include"world.hpp"

void wrap_world(py::module& m) {

      py::class_<World>(m, "World", "TODO: add documentation")

          .def(py::init<real3>(),
               py::arg("cellsize"),
               "construct a World with a given cellsize")

          .def("cellsize",
               &World::cellsize,
               "return the cellsize")

          .def("addFerromagnet",
               &World::addFerromagnet,
               py::arg("name"),
               py::arg("grid"),
               "add a ferromagnet to the world",
               py::return_value_policy::reference);

}