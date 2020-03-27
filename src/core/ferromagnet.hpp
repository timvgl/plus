#pragma once

#include <map>
#include <string>

#include "anisotropy.hpp"
#include "effectivefield.hpp"
#include "exchange.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "quantity.hpp"
#include "system.hpp"
#include "torque.hpp"

class World;

class Ferromagnet : public System {
 public:
  Ferromagnet(World* world, std::string name, Grid grid);
  ~Ferromagnet();
  Ferromagnet(Ferromagnet&&) = default;

  Field* magnetization() const;

  real3 anisU;
  real msat, ku1, aex, alpha;

  const Quantity* anisotropyField() const;
  const Quantity* exchangeField() const;
  const Quantity* effectiveField() const;
  const Quantity* torque() const;

 private:
  Ferromagnet(const Ferromagnet&);
  Ferromagnet& operator=(const Ferromagnet&);

 private:
  Field* magnetization_;

  AnisotropyField anisotropyField_;
  ExchangeField exchangeField_;
  EffectiveField effectiveField_;
  Torque torque_;
};