#include "timesolver.hpp"

#include <memory>

#include "dynamicequation.hpp"
#include "field.hpp"
#include "quantity.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def(py::init([](Variable* x, Quantity* rhs, real timestep) {
        return std::unique_ptr<TimeSolver>(
            new TimeSolver(DynamicEquation(x, rhs), timestep));
      }))
      .def_property_readonly("time", &TimeSolver::time)
      .def_property_readonly("timestep", &TimeSolver::timestep)
      .def("step", &TimeSolver::step);
}