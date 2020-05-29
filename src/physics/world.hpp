#pragma once

#include <map>
#include <memory>

#include "datatypes.hpp"
#include "handler.hpp"

class Ferromagnet;
class Grid;

class World {
 public:
  World(real3 cellsize);
  ~World();
  real3 cellsize() const;
  real cellVolume() const;

  real3 biasMagneticField;

  Ferromagnet* addFerromagnet(Grid grid, std::string name = "");

  // returns a nullptrs if there is no magnet with specified name
  Ferromagnet* getFerromagnet(std::string name) const;

 private:
  std::map<std::string, Ferromagnet*> Ferromagnets;
  real3 cellsize_;
};