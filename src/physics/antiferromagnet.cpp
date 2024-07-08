#include "antiferromagnet.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"

Antiferromagnet::Antiferromagnet(std::shared_ptr<System> system_ptr,
                                 std::string name)
    : Magnet(system_ptr, name),
      afmex_cell(system(), 0.0),
      afmex_nn(system(), 0.0),
      latcon(system(), 0.35e-9),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)) {}
      
Antiferromagnet::Antiferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry)
    : Antiferromagnet(std::make_shared<System>(world, grid, geometry), name) {}

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

void Antiferromagnet::relax() {
  std::vector<real> threshold = {sub1()->RelaxTorqueThreshold.getUniformValue(),
                                 sub2()->RelaxTorqueThreshold.getUniformValue()};
    if (threshold[0] > 0.0 && threshold[1] <= 0.0)
      threshold[1] = threshold[0];
    else if (threshold[0] <= 0.0 && threshold[1] > 0.0)
      threshold[0] == threshold[1];

    Relaxer relaxer(this, threshold);
    relaxer.exec();
}