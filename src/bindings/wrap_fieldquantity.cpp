#include "field.hpp"
#include "fieldquantity.hpp"
#include "ferromagnetquantity.hpp"
#include "wrappers.hpp"

#include<memory>

void wrap_fieldquantity(py::module& m) {
  py::class_<FieldQuantity>(m, "FieldQuantity")
      .def_property_readonly("name", &FieldQuantity::name)
      .def_property_readonly("unit", &FieldQuantity::unit)
      .def_property_readonly("ncomp", &FieldQuantity::ncomp)
      .def_property_readonly("grid", &FieldQuantity::grid)
      .def("eval",
           [](const FieldQuantity* q) { 
             Field f = q->eval();
             return fieldToArray(&f); })
      .def("average", &FieldQuantity::average);
}

void wrap_ferromagnetfieldquantity(py::module& m) {
  py::class_<FM_FieldQuantity, FieldQuantity>(m, "FerromagnetFieldQuantity");
}