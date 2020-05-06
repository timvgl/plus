#include "fieldquantity.hpp"
#include "table.hpp"
#include "wrappers.hpp"

void wrap_table(py::module& m) {
  py::class_<Table>(m, "Table")
      .def(py::init<>())
      .def("add", &Table::addColumn)
      .def("get", &Table::getValues)
      .def("__getitem__", &Table::getValues)
      .def("write_line", &Table::writeLine);
}