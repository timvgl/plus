#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class TotalEnergyDensity: public FerromagnetFieldQuantity {
 public:
  TotalEnergyDensity(Handle<Ferromagnet>);
  void evalIn(Field*) const;
};

class TotalEnergy : public FerromagnetScalarQuantity {
 public:
  TotalEnergy(Handle<Ferromagnet>);
  real eval() const;
};