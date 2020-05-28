#include "ferromagnet.hpp"

#include <random>

#include "fieldquantity.hpp"
#include "magnetfield.hpp"
#include "minimizer.hpp"
#include "world.hpp"

Ferromagnet::Ferromagnet(World* world, std::string name, Grid grid)
    : System(world, name, grid),
      magnetization_(name + ":magnetization", "", 3, grid),
      aex(grid, 0.0),
      msat(grid, 1.0),
      ku1(grid, 0.0),
      alpha(grid, 0.0),
      temperature(grid, 0.0),
      anisU(grid, {0, 0, 0}),
      enableDemag(true) {
  {
    // TODO: this can be done much more efficient somewhere else
    int ncomp = 3;
    int nvalues = ncomp * grid_.ncells();
    std::vector<real> randomValues(nvalues);
    std::uniform_real_distribution<real> unif(-1, 1);
    std::default_random_engine randomEngine;
    for (auto& v : randomValues) {
      v = unif(randomEngine);
    }
    Field randomField(grid_, ncomp);
    randomField.setData(&randomValues[0]);
    magnetization_.set(&randomField);
  }
}

Ferromagnet::~Ferromagnet() {
  for (auto& entry : magnetFields_) {
    delete entry.second;
  }
}

Handle<Ferromagnet> Ferromagnet::getHandle() const {
  return world_->getFerromagnet(name_);
}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(getHandle(), tol, nSamples);
  minimizer.exec();
}

const MagnetField* Ferromagnet::getMagnetField(
    Handle<Ferromagnet> magnet) const {
  auto it = magnetFields_.find(magnet);
  if (it == magnetFields_.end())
    return nullptr;
  return it->second;
}

std::vector<const MagnetField*> Ferromagnet::getMagnetFields() const {
  std::vector<const MagnetField*> magnetFields;
  magnetFields.reserve(magnetFields_.size());
  for (const auto& entry : magnetFields_) {
    magnetFields.push_back(entry.second);
  }
  return magnetFields;
}

void Ferromagnet::addMagnetField(Handle<Ferromagnet> magnet,
                                 MagnetFieldComputationMethod method) {
  if (world_->getFerromagnet(magnet->name()).get() == nullptr) {
    throw std::runtime_error(
        "Can not define the field of the magnet on this magnet because it is "
        "not in the same world.");
  }

  auto it = magnetFields_.find(magnet);
  if (it != magnetFields_.end()) {
    // MagnetField is already registered, just need to update the method
    it->second->setMethod(method);
    return;
  }

  magnetFields_[magnet] = new MagnetField(magnet, grid_, method);
}

void Ferromagnet::removeMagnetField(Handle<Ferromagnet> magnet) {
  auto it = magnetFields_.find(magnet);
  if (it != magnetFields_.end()) {
    delete it->second;
    magnetFields_.erase(it);
  }
}