#pragma once

#include "datatypes.hpp"
#include "grid.hpp"

class Field;

class DemagKernel {
 public:
  DemagKernel(Grid dst, Grid src, real3 cellsize);
  ~DemagKernel();

  Grid grid() const;
  real3 cellsize() const;
  const Field* field() const;

  void compute();

 private:
  Grid grid_;
  Grid dstGrid_;
  Grid srcGrid_;
  real3 cellsize_;
  Field* kernel_;

 public:
  // Helper function which determines the kernel grid
  static Grid kernelGrid(Grid dst, Grid src);
};