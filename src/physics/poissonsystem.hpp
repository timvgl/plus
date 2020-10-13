#pragma once

#include <stdexcept>

#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"

class PoissonSystem {
 public:
  PoissonSystem(const Ferromagnet* magnet) : magnet_(magnet) {}

  Grid grid() const { return magnet_->grid(); }
  const Ferromagnet* magnet() const { return magnet_; }

  void construct();
  Field apply(const Field&);
  Field solve();

 private:
  const Ferromagnet* magnet_;
  Field matrixValues_;
  Field rhs_;
};