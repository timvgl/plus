#include "antiferromagnet.hpp"

#include <algorithm>
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
      biasMagneticField(system(), {0, 0, 0}),
      sub1_(Ferromagnet(world, grid, name + ":sublattice_1", geometry)),
      sub2_(Ferromagnet(world, grid, name + ":sublattice_2", geometry)) {}

const Ferromagnet* Antiferromagnet::sub1() const {
  return &sub1_;
}

const Ferromagnet* Antiferromagnet::sub2() const {
  return &sub2_;
}

const Ferromagnet* Antiferromagnet::getOtherSublattice(const Ferromagnet* sub) const {
  if (std::find(sublattices_.begin(), sublattices_.end(), sub) == sublattices_.end())
    throw std::out_of_range("Sublattice not found in Antiferromagnet.");
  return sublattices_[0] == sub ? &sub2_ : &sub1_;
}

std::vector<const Ferromagnet*> Antiferromagnet::sublattices() const {
  return sublattices_;
}