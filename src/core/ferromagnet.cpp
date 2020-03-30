#include "ferromagnet.hpp"

Ferromagnet::Ferromagnet(World* world, std::string name, Grid grid)
    : System(world, name, grid),
      anisotropyField_(this),
      exchangeField_(this),
      effectiveField_(this),
      torque_(this),
      magnetization_(name + ":magnetization", "", 3, grid) {}

Ferromagnet::~Ferromagnet() {}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const Quantity* Ferromagnet::anisotropyField() const {
  return &anisotropyField_;
}

const Quantity* Ferromagnet::exchangeField() const {
  return &exchangeField_;
}

const Quantity* Ferromagnet::effectiveField() const {
  return &effectiveField_;
}

const Quantity* Ferromagnet::torque() const {
  return &torque_;
}