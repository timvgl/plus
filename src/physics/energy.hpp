#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class TotalEnergyDensity: public FerromagnetFieldQuantity {
 public:
  TotalEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
};

class TotalEnergy : public FerromagnetScalarQuantity {
 public:
  TotalEnergy(Ferromagnet*);
  real eval() const;
};