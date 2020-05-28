#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class ExternalField : public FerromagnetFieldQuantity {
 public:
  ExternalField(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ZeemanEnergyDensity : public FerromagnetFieldQuantity {
 public:
  ZeemanEnergyDensity(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ZeemanEnergy : public FerromagnetScalarQuantity {
 public:
  ZeemanEnergy(Handle<Ferromagnet>);
  real eval() const;
};