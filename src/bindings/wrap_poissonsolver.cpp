#include "poissonsolver.hpp"
#include "wrappers.hpp"

void wrap_poissonsolver(py::module& m) {
  py::class_<PoissonSolver>(m, "PoissonSolver")
      .def("step", &PoissonSolver::step)
      .def("init", &PoissonSolver::init)
      .def("solve", [](PoissonSolver* p) { return fieldToArray(p->solve()); })
      .def("state",
           [](const PoissonSolver* p) { return fieldToArray(p->state()); });
}
