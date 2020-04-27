#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

class AnisotropyField : public FerromagnetFieldQuantity {
 public:
  AnisotropyField(Ferromagnet*);
  void evalIn(Field*) const;
};