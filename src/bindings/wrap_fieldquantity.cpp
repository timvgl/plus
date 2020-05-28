#include "field.hpp"
#include "fieldquantity.hpp"
#include "wrappers.hpp"

#include<memory>

void wrap_fieldquantity(py::module& m) {
  py::class_<FieldQuantity>(m, "FieldQuantity")
      .def_property_readonly("name", &FieldQuantity::name)
      .def_property_readonly("unit", &FieldQuantity::unit)
      .def_property_readonly("ncomp", &FieldQuantity::ncomp)
      .def_property_readonly("grid", &FieldQuantity::grid)
      .def("eval",
           [](const FieldQuantity* q) { return fieldToArray(q->eval().get()); })
      .def("average", &FieldQuantity::average);
}