#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class ExchangeField : public FerromagnetFieldQuantity {
 public:
  ExchangeField(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ExchangeEnergyDensity : public FerromagnetFieldQuantity {
 public:
  ExchangeEnergyDensity(Handle<Ferromagnet>);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class ExchangeEnergy : public FerromagnetScalarQuantity {
 public:
  ExchangeEnergy(Handle<Ferromagnet>);
  real eval() const;
};