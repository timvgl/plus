#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class Torque : public FerromagnetQuantity {
 public:
  Torque(Ferromagnet*);
  void evalIn(Field*) const;
};