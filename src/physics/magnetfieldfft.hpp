#pragma once

#include <cufft.h>

#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"
#include "magnetfield.hpp"
#include "magnetfieldkernel.hpp"

class Parameter;
class Field;

class MagnetFieldFFTExecutor : public MagnetFieldExecutor {
 public:
  MagnetFieldFFTExecutor(Grid grid, real3 cellsize);
  ~MagnetFieldFFTExecutor();
  void exec(Field* h, const Field* m, const Parameter* msat) const;

 private:
  Grid grid_;
  real3 cellsize_;
  MagnetFieldKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
