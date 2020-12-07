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
  LinSolver* solver();

 private:
  std::unique_ptr<LinearSystem> construct() const;
  std::unique_ptr<LinearSystem> construct_isotropic() const;

 private:
  const Ferromagnet* magnet_;
  std::unique_ptr<LinSolver> solver_;
};
