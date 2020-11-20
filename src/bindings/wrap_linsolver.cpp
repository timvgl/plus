#include "linsolver.hpp"
#include "wrappers.hpp"

void wrap_linsolver(py::module& m) {
  py::class_<LinSolver>(m, "LinSolver")
      .def("step", &LinSolver::step)
      .def("restart", &LinSolver::restartStepper)
      .def("max_norm_residual", &LinSolver::residualMaxNorm)
      .def("set_method", &LinSolver::setMethodByName)
      .def("solve", [](LinSolver* p) { return fieldToArray(p->solve()); })
      .def_readwrite("max_iter", &LinSolver::maxIterations)
      .def_readwrite("tol", &LinSolver::tol)
      .def("state",
           [](const LinSolver* p) { return fieldToArray(p->getState()); });
}
