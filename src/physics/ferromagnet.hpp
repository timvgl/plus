#pragma once

#include <map>
#include <string>

#include "anisotropy.hpp"
#include "demag.hpp"
#include "effectivefield.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "quantity.hpp"
#include "system.hpp"
#include "torque.hpp"
#include "variable.hpp"

class World;

class Ferromagnet : public System {
 public:
  Ferromagnet(World* world, std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;

  const Variable* magnetization() const;

  real msat, aex, alpha;
  bool enableDemag;

  VectorParameter anisU;
  Parameter ku1;

  const Quantity* demagField() const;
  const Quantity* anisotropyField() const;
  const Quantity* exchangeField() const;
  const Quantity* effectiveField() const;
  const Quantity* torque() const;

  void minimize(real tol = 1e-6, int nSamples = 10);

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  NormalizedVariable magnetization_;

  DemagField demagField_;
  AnisotropyField anisotropyField_;
  ExchangeField exchangeField_;
  EffectiveField effectiveField_;
  Torque torque_;
};