#pragma once

#include <vector>

#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"

class World {
 public:
  World(real3 cellsize);
  ~World();
  real3 cellsize() const;

  real3 biasMagneticField;

  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");

 private:
  std::vector<Ferromagnet> Ferromagnets;
  real3 cellsize_;
};