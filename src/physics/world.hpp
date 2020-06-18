#pragma once

#include <map>
#include <memory>
#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"

class Ferromagnet;

class World {
 public:
  World(real3 cellsize, Grid mastergrid = Grid(int3{0, 0, 0}));
  ~World();
  real3 cellsize() const;
  real cellVolume() const;
  Grid mastergrid() const;

  /// Returns true if grid is completely inside the mastergrid
  bool inMastergrid(Grid) const;

  real3 biasMagneticField;

  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");

  // returns a nullptrs if there is no magnet with specified name
  Ferromagnet* getFerromagnet(std::string name) const;

 private:
  std::map<std::string, Ferromagnet*> Ferromagnets;
  real3 cellsize_;
  Grid mastergrid_;
};