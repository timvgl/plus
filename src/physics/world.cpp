#include "world.hpp"

#include <stdexcept>
#include <string>

World::World(real3 cellsize) : cellsize_(cellsize) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
};

World::~World() {}

real3 World::cellsize() const {
  return cellsize_;
}

Ferromagnet* World::addFerromagnet(Grid grid, std::string name) {
  if (Ferromagnets.size() > 0) {
    throw std::out_of_range(
        "Having a world with multiple ferromagnets is not yet possible");
  }
  if (name.length()==0) {
    name = "magnet_1";
  }
  Ferromagnets.emplace_back(this, name, grid);
  return &Ferromagnets.back();
};