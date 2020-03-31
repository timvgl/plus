#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class DemagField : public FerromagnetQuantity {
 public:
  DemagField(Ferromagnet*);
  void evalIn(Field*) const;
};