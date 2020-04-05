#pragma once

#include "ferromagnetquantity.hpp"
#include "demagkernel.hpp"
#include "demagconvolution.hpp"

class Ferromagnet;
class Field;

class DemagField : public FerromagnetQuantity {
 public:
  DemagField(Ferromagnet*);
  void evalIn(Field*) const;
 private:
  DemagKernel demagkernel_;
  DemagConvolution convolution_;
};