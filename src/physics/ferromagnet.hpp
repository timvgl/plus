#pragma once

#include <map>
#include <string>
#include <vector>

#include "anisotropy.hpp"
#include "demag.hpp"
#include "effectivefield.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "magnetfield.hpp"
#include "parameter.hpp"
#include "system.hpp"
#include "thermalnoise.hpp"
#include "torque.hpp"
#include "variable.hpp"
#include "zeeman.hpp"

class FieldQuantity;
class World;

class Ferromagnet : public System {
 public:
  Ferromagnet(World* world, std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;

  const Variable* magnetization() const;

  bool enableDemag;

  VectorParameter anisU;
  Parameter msat, aex, ku1, alpha, temperature;

  const FieldQuantity* demagField() const;
  const FieldQuantity* demagEnergyDensity() const;
  const ScalarQuantity* demagEnergy() const;

  const FieldQuantity* externalField() const;
  const FieldQuantity* zeemanEnergyDensity() const;
  const ScalarQuantity* zeemanEnergy() const;

  const FieldQuantity* anisotropyField() const;
  const FieldQuantity* anisotropyEnergyDensity() const;
  const ScalarQuantity* anisotropyEnergy() const;

  const FieldQuantity* exchangeField() const;
  const FieldQuantity* exchangeEnergyDensity() const;
  const ScalarQuantity* exchangeEnergy() const;

  const FieldQuantity* effectiveField() const;
  const FieldQuantity* totalEnergyDensity() const;
  const ScalarQuantity* totalEnergy() const;

  const FieldQuantity* thermalNoise() const;

  const FieldQuantity* torque() const;

  const MagnetField* getMagnetField(Ferromagnet*) const;
  std::vector<const MagnetField*> getMagnetFields() const;
  void addMagnetField(
      Ferromagnet*,
      MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_BRUTE);
  void removeMagnetField(Ferromagnet*);

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  NormalizedVariable magnetization_;

  DemagField demagField_;
  DemagEnergyDensity demagEnergyDensity_;
  DemagEnergy demagEnergy_;

  ExternalField externalField_;
  ZeemanEnergyDensity zeemanEnergyDensity_;
  ZeemanEnergy zeemanEnergy_;

  AnisotropyField anisotropyField_;
  AnisotropyEnergyDensity anisotropyEnergyDensity_;
  AnisotropyEnergy anisotropyEnergy_;

  ExchangeField exchangeField_;
  ExchangeEnergyDensity exchangeEnergyDensity_;
  ExchangeEnergy exchangeEnergy_;

  EffectiveField effectiveField_;
  TotalEnergyDensity totalEnergyDensity_;
  TotalEnergy totalEnergy_;

  ThermalNoise thermalNoise_;

  Torque torque_;

  std::map<Ferromagnet*, MagnetField*> magnetFields_;
};