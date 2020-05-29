#include "scalarquantity.hpp"
#include "ferromagnetquantity.hpp"
#include "wrappers.hpp"

void wrap_scalarquantity(py::module& m) {
  py::class_<ScalarQuantity>(m, "ScalarQuantity")
      .def_property_readonly("name", &ScalarQuantity::name)
      .def_property_readonly("unit", &ScalarQuantity::unit)
      .def("eval", &ScalarQuantity::eval);
}

void wrap_ferromagnetscalarquantity(py::module& m) {
  py::class_<FM_ScalarQuantity, ScalarQuantity>(m, "FerromagnetScalarQuantity");
}