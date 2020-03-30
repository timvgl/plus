#include "variable.hpp"

#include <memory>

#include "field.hpp"
#include "wrappers.hpp"

void wrap_variable(py::module& m) {
  py::class_<Variable>(m, "Variable")
      .def_property_readonly("name", &Variable::name)
      .def_property_readonly("unit", &Variable::unit)
      .def_property_readonly("ncomp", &Variable::ncomp)
      .def_property_readonly("grid", &Variable::grid)
      .def("get", [](const Variable* v) { return fieldToArray(v->field()); })
      .def("set", [](const Variable* v, py::array_t<real> data) {
        std::unique_ptr<Field> tmp(new Field(v->grid(), v->ncomp()));
        setArrayInField(tmp.get(), data);
        v->set(tmp.get());
      });
}