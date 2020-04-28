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
  bool assuredZero() const override;
 private:
  DemagKernel demagkernel_;
  DemagConvolution convolution_;
};

class DemagEnergyDensity : public FerromagnetFieldQuantity {
 public:
  DemagEnergyDensity(Ferromagnet*);
  void evalIn(Field*) const;
  bool assuredZero() const override;
};

class DemagEnergy : public FerromagnetScalarQuantity {
 public:
  DemagEnergy(Ferromagnet*);
  real eval() const;
};