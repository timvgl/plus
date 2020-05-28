#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class EffectiveField : public FerromagnetFieldQuantity {
 public:
  EffectiveField(Handle<Ferromagnet>);
  void evalIn(Field*) const;
};