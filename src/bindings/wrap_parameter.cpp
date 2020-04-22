#include <memory>

#include "field.hpp"
#include "parameter.hpp"
#include "quantity.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  py::class_<Parameter, Quantity>(m, "Parameter")
      .def("is_uniform", &Parameter::isUniform)
      .def("set", [](Parameter* p, real value) { p->set(value); })
      .def("set", [](Parameter* p, py::array_t<real> data) {
        std::unique_ptr<Field> tmp(new Field(p->grid(), 1));
        setArrayInField(tmp.get(), data);
        p->set(tmp.get());  // TODO: check if this can be done without an extra
                            // copy
      });
}