#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class AnisotropyField : public FerromagnetFieldQuantity {
 public:
  AnisotropyField(Ferromagnet*);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class AnisotropyEnergyDensity : public FerromagnetFieldQuantity {
 public:
  AnisotropyEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class AnisotropyEnergy : public FerromagnetScalarQuantity {
 public:
  AnisotropyEnergy(Ferromagnet*);
  real eval() const;
};