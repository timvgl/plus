#pragma once

#include "gpubuffer.hpp"
#include "vec.hpp"

class CuLinearSystem;

class LinearSystem {
  int nnz_;
  int nrows_;
  GVec b_;
  std::vector<GVec> matrixval_;
  GpuBuffer<lsReal*> matrixvalPtrs_;
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
  lsReal** a;
  lsReal* b;

  CuLinearSystem(int nrows,
                 int nNonZerosInRow,
                 int** idx,
                 lsReal** a,
                 lsReal* b)
      : nrows(nrows), nnz(nNonZerosInRow), idx(idx), a(a), b(b) {}
};
