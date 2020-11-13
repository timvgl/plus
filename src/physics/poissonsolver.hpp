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

  void init();
  Field solve();
  void step();

  Field state() const;
  Field residual() const;
  real residualMaxNorm() const;

  int maxIterations;
  double tol;

 private:
  const Ferromagnet* magnet_;
  LinearSystem sys_;
  Field pot_;
  std::unique_ptr<LinearSystemSolverStepper> stepper_;
  int nstep_;
};