#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class EffectiveField : public FerromagnetFieldQuantity {
 public:
  EffectiveField(Ferromagnet*);
  void evalIn(Field*) const;
};