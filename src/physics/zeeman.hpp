#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class ExternalField : public FerromagnetFieldQuantity {
 public:
  ExternalField(Ferromagnet*);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ZeemanEnergyDensity : public FerromagnetFieldQuantity {
 public:
  ZeemanEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ZeemanEnergy : public FerromagnetScalarQuantity {
 public:
  ZeemanEnergy(Ferromagnet*);
  real eval() const;
};