#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class AnisotropyField : public FerromagnetQuantity {
 public:
  AnisotropyField(Ferromagnet*);
  void evalIn(Field*) const;
};