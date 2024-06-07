#include "antiferromagnet.hpp"

#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "mumaxworld.hpp"

#include "system.hpp"

Antiferromagnet::Antiferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : Magnet(world, grid, name, geometry),
      afmex_cell(system(), 0.0),
      afmex_nn(system(), 0.0),
      latcon(system(), 0.35e-9),
      sub1(Ferromagnet(world, grid, name + ":sublattice_1", geometry)),
      sub2(Ferromagnet(world, grid, name + ":sublattice_2", geometry)) {}

std::vector<Ferromagnet*> Antiferromagnet::sublattices() const {
  return sublattices_;
}