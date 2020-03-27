#include "world.hpp"

#include <stdexcept>

World::World(real3 cellsize) : cellsize_(cellsize) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }
};

World::~World() {}

real3 World::cellsize() const {
  return cellsize_;
}

Ferromagnet* World::addFerromagnet(std::string name, Grid grid) {
  if (Ferromagnets.size() > 0) {
    throw std::out_of_range(
        "Having a world with multiple ferromagnets is not yet possible");
  }
  Ferromagnets.emplace_back(this, name, grid);
  return &Ferromagnets.back();
};