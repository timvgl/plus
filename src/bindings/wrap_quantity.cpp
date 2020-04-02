#include "field.hpp"
#include "quantity.hpp"
#include "wrappers.hpp"

void wrap_quantity(py::module& m) {
  py::class_<Quantity>(m, "Quantity")
      .def_property_readonly("name", &Quantity::name)
      .def_property_readonly("unit", &Quantity::unit)
      .def_property_readonly("ncomp", &Quantity::ncomp)
      .def_property_readonly("grid", &Quantity::grid)
      .def("eval",
           [](const Quantity* q) { return fieldToArray(q->eval().get()); });
}