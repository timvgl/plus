#include <memory>
#include <stdexcept>

#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_magnet(py::module& m) {
  py::class_<Magnet>(m, "Magnet")
      .def_property_readonly("name", &Magnet::name)
      .def_property_readonly("system", &Magnet::system)
      .def_property_readonly("world", &Magnet::mumaxWorld);
}
