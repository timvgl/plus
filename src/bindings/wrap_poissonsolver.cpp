#include "poissonsolver.hpp"
#include "wrappers.hpp"

void wrap_poissonsolver(py::module& m) {
  py::class_<PoissonSolver>(m, "PoissonSolver")
      .def("step", &PoissonSolver::step)
      .def("init", &PoissonSolver::init)
      .def("set_method", &PoissonSolver::setMethodByName)
      .def("restart", &PoissonSolver::restart)
      .def("max_norm_residual", &PoissonSolver::residualMaxNorm)
      .def("solve", [](PoissonSolver* p) { return fieldToArray(p->solve()); })
      .def_readwrite("max_iter", &PoissonSolver::maxIterations)
      .def_readwrite("tol", &PoissonSolver::tol)
      .def("state",
           [](const PoissonSolver* p) { return fieldToArray(p->state()); });
}
