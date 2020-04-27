#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class Torque : public FerromagnetFieldQuantity {
 public:
  Torque(Ferromagnet*);
  void evalIn(Field*) const;
};

class RelaxTorque : public FerromagnetFieldQuantity {
 public:
  RelaxTorque(Ferromagnet*);
  void evalIn(Field*) const;
};