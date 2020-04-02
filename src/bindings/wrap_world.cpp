#include "world.hpp"
#include "wrappers.hpp"



void wrap_world(py::module& m) {
  // TODO: avoid destructor being called when python ref out of scope
  py::class_<World>(m, "World", "TODO: add documentation")

      .def(py::init<real3>(), py::arg("cellsize"),
           "construct a World with a given cellsize")
      .def_property_readonly("cellsize", &World::cellsize, "the cellsize of the world")
      .def("add_ferromagnet", &World::addFerromagnet, py::arg("name"),
           py::arg("grid"), "add a ferromagnet to the world",
           py::return_value_policy::reference);
}