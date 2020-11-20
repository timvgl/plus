#pragma once

#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"

const int NNEAREST = 5;  // number of relevant nearest neighbors

class Ferromagnet;

class PoissonSystem {
 public:
  PoissonSystem(const Ferromagnet* magnet);

  void init();
  Field solve();
  LinSolver* solver();

 private:
  std::unique_ptr<LinearSystem> construct() const;

 private:
  const Ferromagnet* magnet_;
  std::unique_ptr<LinSolver> solver_;
};