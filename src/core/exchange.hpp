#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class ExchangeField : public FerromagnetQuantity {
 public:
  ExchangeField(Ferromagnet*);
  void evalIn(Field*) const;
};