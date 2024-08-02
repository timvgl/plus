#include "world.hpp"

#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"
#include "timesolver.hpp"

World::World(real3 cellsize, Grid mastergrid)
    : cellsize_(cellsize),
      mastergrid_(mastergrid),
      timesolver_(TimeSolver::Factory::create()) {
  if (cellsize.x <= 0 || cellsize.y <= 0 || cellsize.z <= 0) {
    throw std::invalid_argument("The cell size should be larger than 0");
  }

  // TODO: move this code and make user accessible! This does not belong here!
  pbcRepetitions_ = int3{0,0,0};
  int repeat = 4;
  if (this->mastergrid().size().x > 0)
    pbcRepetitions_.x = repeat;
  if (this->mastergrid().size().y > 0)
    pbcRepetitions_.y = repeat;
  if (this->mastergrid().size().z > 0)
    pbcRepetitions_.z = repeat;
}

World::~World() {}

real World::time() const {
  return timesolver_->time();
}

real3 World::cellsize() const {
  return cellsize_;
}

real World::cellVolume() const {
  return cellsize_.x * cellsize_.y * cellsize_.z;
}

Grid World::mastergrid() const {
  return mastergrid_;
}

const int3 World::pbcRepetitions() const {
  return pbcRepetitions_;
}

bool World::inMastergrid(Grid grid) const {
  int3 d1 = grid.origin() - mastergrid_.origin();
  int3 d2 = d1 + grid.size() - mastergrid_.size();
  if (mastergrid_.size().x > 0 && (d1.x < 0 || d2.x > 0))
    return false;
  if (mastergrid_.size().y > 0 && (d1.y < 0 || d2.y > 0))
    return false;
  if (mastergrid_.size().z > 0 && (d1.z < 0 || d2.z > 0))
    return false;
  return true;
}

TimeSolver& World::timesolver() const {
  return *timesolver_;
}
