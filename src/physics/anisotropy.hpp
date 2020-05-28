#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class AnisotropyField : public FerromagnetFieldQuantity {
 public:
  AnisotropyField(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class AnisotropyEnergyDensity : public FerromagnetFieldQuantity {
 public:
  AnisotropyEnergyDensity(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class AnisotropyEnergy : public FerromagnetScalarQuantity {
 public:
  AnisotropyEnergy(Handle<Ferromagnet>);
  real eval() const;
};