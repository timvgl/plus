#pragma once

#include "ferromagnetquantity.hpp"
#include "scalarquantity.hpp"

class Ferromagnet;
class Field;

class AnisotropyField : public FerromagnetFieldQuantity {
 public:
  AnisotropyField(Ferromagnet*);
  void evalIn(Field*) const;
};

class AnisotropyEnergyDensity : public FerromagnetFieldQuantity {
 public:
  AnisotropyEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
};

class AnisotropyEnergy : public FerromagnetScalarQuantity {
 public:
  AnisotropyEnergy(Ferromagnet*);
  real eval() const;
};