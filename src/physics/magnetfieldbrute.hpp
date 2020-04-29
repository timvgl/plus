#pragma once

#include "magnetfieldkernel.hpp"
#include "magnetfield.hpp"

class MagnetFieldBruteExecutor : public MagnetFieldExecutor {
 public:
  MagnetFieldBruteExecutor(Grid grid, real3 cellsize);
  void exec(Field* h, const Field* m, const Parameter* msat) const;

 private:
  MagnetFieldKernel kernel_;
};
