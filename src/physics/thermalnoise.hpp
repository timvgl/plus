#pragma once

#include "curand.h"
#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

class ThermalNoise : public FerromagnetFieldQuantity {
 public:
  ThermalNoise(Handle<Ferromagnet>);
  ~ThermalNoise();
  void evalIn(Field*) const;
  bool assuredZero() const override;

 private:
  curandGenerator_t generator_;
};
