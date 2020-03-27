#include "system.hpp"

System::System(World* world, std::string name, Grid grid)
    : grid_(grid), name_(name), world_(world) {}

World* System::world() const {
  return world_;
}

std::string System::name() const {
  return name_;
}

Grid System::grid() const {
  return grid_;
}