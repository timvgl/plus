#include "magnet.hpp"

#include <curand.h>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "poissonsystem.hpp"
#include "strayfield.hpp"
#include "system.hpp"

Magnet::Magnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : system_(new System(world, grid, geometry)) {}

std::string Magnet::name() const {
  return name_;
}

std::shared_ptr<const System> Magnet::system() const {
  return system_;
}

const World* Magnet::world() const {
  return system()->world();
}

Grid Magnet::grid() const {
  return system()->grid();
}

real3 Magnet::cellsize() const {
  return world()->cellsize();
}

const GpuBuffer<bool>& Magnet::getGeometry() const {
  return system_->geometry();
}


