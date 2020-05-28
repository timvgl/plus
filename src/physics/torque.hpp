#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class Torque : public FerromagnetFieldQuantity {
 public:
  Torque(Handle<Ferromagnet>);
  void evalIn(Field*) const;
};

class RelaxTorque : public FerromagnetFieldQuantity {
 public:
  RelaxTorque(Handle<Ferromagnet>);
  void evalIn(Field*) const;
};