#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class ExchangeField : public FerromagnetFieldQuantity {
 public:
  ExchangeField(Ferromagnet*);
  void evalIn(Field*) const;
};