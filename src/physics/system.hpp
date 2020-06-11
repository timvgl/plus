#pragma once

#include <string>

#include "grid.hpp"

class World;

class System {
 public:
  System(World* world, std::string name, Grid grid);

  World* world() const;
  std::string name() const;
  Grid grid() const;
  real3 cellsize() const;

 protected:
  World* world_;
  std::string name_;
  Grid grid_;
};