#pragma once

#include <stdexcept>

#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "linsystem.hpp"

const int NNEAREST = 5;  // number of relevant nearest neighbors

class PoissonSystem {
 public:
  PoissonSystem(const Ferromagnet* magnet)
      : magnet_(magnet), sys_(magnet->grid(), NNEAREST) {}

  Grid grid() const { return magnet_->grid(); }
  const Ferromagnet* magnet() const { return magnet_; }

  void construct();
  Field solve();

 private:
  const Ferromagnet* magnet_;
  LinearSystem sys_;
};