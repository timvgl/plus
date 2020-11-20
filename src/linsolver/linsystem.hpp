#pragma once

#include "datatypes.hpp"
#include "gpubuffer.hpp"
#include "vec.hpp"

class CuLinearSystem;

class LinearSystem {
  int nnz_;
  int nrows_;
  GVec b_;
  std::vector<GVec> matrixval_;
  GpuBuffer<real*> matrixvalPtrs_;
  std::vector<GpuBuffer<int>> matrixidx_;
  GpuBuffer<int*> matrixidxPtrs_;

 public:
  LinearSystem(int nrows, int nonZerosInRow);

  int nrows() const { return nrows_; }
  GVec matrixmul(const GVec& x) const;
  GVec residual(const GVec& x) const;

  CuLinearSystem cu() const;

 private:
  void allocate();
  void free();
};

struct CuLinearSystem {
  const int nrows;
  const int nnz;
  int** idx;
  real** a;
  real* b;

  CuLinearSystem(int nrows, int nNonZerosInRow, int** idx, real** a, real* b)
      : nrows(nrows), nnz(nNonZerosInRow), idx(idx), a(a), b(b) {}
};
