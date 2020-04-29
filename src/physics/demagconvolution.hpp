#pragma once

#include <cufft.h>

#include <vector>

#include "datatypes.hpp"
#include "demagkernel.hpp"
#include "grid.hpp"

class Parameter;

class Field;

class DemagConvolution {
 public:
  DemagConvolution(Grid grid, real3 cellsize);
  ~DemagConvolution();
  void exec(Field* h, const Field* m, Parameter* msat) const;

 private:
  Grid grid_;
  real3 cellsize_;
  DemagKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
