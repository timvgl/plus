#pragma once

#include "ferromagnetquantity.hpp"
#include "magnetfield.hpp"
#include "magnetfieldkernel.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class DemagField : public FerromagnetFieldQuantity {
 public:
  DemagField(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class DemagEnergyDensity : public FerromagnetFieldQuantity {
 public:
  DemagEnergyDensity(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class DemagEnergy : public FerromagnetScalarQuantity {
 public:
  DemagEnergy(Handle<Ferromagnet>);
  real eval() const;
};