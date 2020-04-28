#pragma once

#include "ferromagnetquantity.hpp"
#include "demagkernel.hpp"
#include "demagconvolution.hpp"

class Ferromagnet;
class Field;

class DemagField : public FerromagnetFieldQuantity {
 public:
  DemagField(Ferromagnet*);
  void evalIn(Field*) const;
 private:
  DemagKernel demagkernel_;
  DemagConvolution convolution_;
};

class DemagEnergyDensity : public FerromagnetFieldQuantity {
 public:
  DemagEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
};

class DemagEnergy : public FerromagnetScalarQuantity {
 public:
  DemagEnergy(Ferromagnet*);
  real eval() const;
};