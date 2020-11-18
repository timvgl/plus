#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"

class Field;
class CuLinearSystem;

class LinearSystem {
  int nnz_;
  Grid grid_;
  GpuBuffer<real> b_;
  std::vector<GpuBuffer<real>> matrixval_;
  GpuBuffer<real*> matrixvalPtrs_;
  std::vector<GpuBuffer<int>> matrixidx_;
  GpuBuffer<int*> matrixidxPtrs_;

 public:
  LinearSystem(Grid grid, int nonZerosInRow);

  Grid grid() const { return grid_; }
  Field matrixmul(const Field& x) const;
  Field residual(const Field& x) const;

  CuLinearSystem cu() const;

 private:
  void allocate();
  void free();
};

struct CuLinearSystem {
  const Grid grid;
  const int nnz;
  int** idx;
  real** a;
  real* b;

  CuLinearSystem(Grid grid, int nNonZerosInRow, int** idx, real** a, real* b)
      : grid(grid), nnz(nNonZerosInRow), idx(idx), a(a), b(b) {}
};
