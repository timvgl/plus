#pragma once

#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"

class Ferromagnet;

class PoissonSystem {
 public:
  explicit PoissonSystem(const Ferromagnet* magnet);

  void init();
  Field solve();

  LinSolver* solver() { return &solver_; }

 private:
  LinearSystem construct() const;

 private:
  const Ferromagnet* magnet_;
  LinSolver solver_;
};
