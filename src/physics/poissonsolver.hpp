#pragma once

#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "linstepper.hpp"
#include "linsystem.hpp"

const int NNEAREST = 5;  // number of relevant nearest neighbors

class Ferromagnet;

class PoissonSolver {
 public:
  using Method = LinearSystemSolverStepper::Method;

 public:
  PoissonSolver(const Ferromagnet* magnet);

  void init();
  Field solve();
  void step();
  void restart();
  void setMethod(Method);
  void setMethodByName(std::string);

  Field state() const;
  Field residual() const;
  real residualMaxNorm() const;

  int maxIterations = -1;
  double tol = 1e-5;

 private:
  const Ferromagnet* magnet_;
  LinearSystem sys_;
  Field pot_;
  std::unique_ptr<LinearSystemSolverStepper> stepper_;
  int nstep_;
};