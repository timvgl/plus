#pragma once

#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"

const int NNEAREST = 5;  // number of relevant nearest neighbors

class Ferromagnet;

class PoissonSolver {
 public:
  PoissonSolver(const Ferromagnet* magnet);
  void construct();
  Field solve();

 private:
  const Ferromagnet* magnet_;
  LinearSystem sys_;
  Field pot_;
  std::unique_ptr<LinearSystemSolverStepper> stepper_;
};