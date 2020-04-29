#include "ferromagnet.hpp"

#include <random>

#include "fieldquantity.hpp"
#include "minimizer.hpp"

Ferromagnet::Ferromagnet(World* world, std::string name, Grid grid)
    : System(world, name, grid),
      demagField_(this),
      demagEnergyDensity_(this),
      demagEnergy_(this),
      externalField_(this),
      zeemanEnergyDensity_(this),
      zeemanEnergy_(this),
      anisotropyField_(this),
      anisotropyEnergyDensity_(this),
      anisotropyEnergy_(this),
      exchangeField_(this),
      exchangeEnergyDensity_(this),
      exchangeEnergy_(this),
      effectiveField_(this),
      totalEnergyDensity_(this),
      totalEnergy_(this),
      torque_(this),
      magnetization_(name + ":magnetization", "", 3, grid),
      aex(grid, 0.0),
      msat(grid, 1.0),
      ku1(grid, 0.0),
      alpha(grid, 0.0),
      anisU(grid, {0, 0, 0}) {
  enableDemag = true;
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

Ferromagnet::~Ferromagnet() {}

const Variable* Ferromagnet::magnetization() const {
  return &magnetization_;
}

const FieldQuantity* Ferromagnet::demagField() const {
  return &demagField_;
}

const FieldQuantity* Ferromagnet::demagEnergyDensity() const {
  return &demagEnergyDensity_;
}

const ScalarQuantity* Ferromagnet::demagEnergy() const {
  return &demagEnergy_;
}

const FieldQuantity* Ferromagnet::externalField() const {
  return &externalField_;
}

const FieldQuantity* Ferromagnet::zeemanEnergyDensity() const {
  return &zeemanEnergyDensity_;
}

const ScalarQuantity* Ferromagnet::zeemanEnergy() const {
  return &zeemanEnergy_;
}

const FieldQuantity* Ferromagnet::anisotropyField() const {
  return &anisotropyField_;
}

const FieldQuantity* Ferromagnet::anisotropyEnergyDensity() const {
  return &anisotropyEnergyDensity_;
}

const ScalarQuantity* Ferromagnet::anisotropyEnergy() const {
  return &anisotropyEnergy_;
}

const FieldQuantity* Ferromagnet::exchangeField() const {
  return &exchangeField_;
}

const FieldQuantity* Ferromagnet::exchangeEnergyDensity() const {
  return &exchangeEnergyDensity_;
}

const ScalarQuantity* Ferromagnet::exchangeEnergy() const {
  return &exchangeEnergy_;
}

const FieldQuantity* Ferromagnet::effectiveField() const {
  return &effectiveField_;
}

const FieldQuantity* Ferromagnet::totalEnergyDensity() const {
  return &totalEnergyDensity_;
}

const ScalarQuantity* Ferromagnet::totalEnergy() const {
  return &totalEnergy_;
}

const FieldQuantity* Ferromagnet::torque() const {
  return &torque_;
}

void Ferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}