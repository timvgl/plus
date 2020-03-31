#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class EffectiveField : public FerromagnetQuantity {
 public:
  EffectiveField(Ferromagnet*);
  void evalIn(Field*) const;
};