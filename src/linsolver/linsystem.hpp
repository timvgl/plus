#pragma once

#include <vector>

#include "gpubuffer.hpp"
#include "vec.hpp"

/** Represent a sparse system of linear equations Ax = b. */
class LinearSystem {
 public:
  LinearSystem();
  LinearSystem(int nrows, int nonZerosInRow);
  LinearSystem(const LinearSystem&);
  LinearSystem(LinearSystem&&);

  LinearSystem& operator=(const LinearSystem&);
  LinearSystem& operator=(LinearSystem&&);

  void clear(); /** Make linear system empty (nRows=0). */

  int nRows() const { return b_.size(); }
  int nNonZerosInMatrixRow() const { return matrixval_.size(); }
  bool empty() const { return nRows() == 0; }

  GVec matrixmul(const GVec& x) const; /** Return Ax */
  GVec residual(const GVec& x) const;  /** Return b - Ax */

  struct CuData {
    int nrows;
    int nnz;
    int** idx;
    lsReal** a;
    lsReal* b;
  };

  CuData cu() const;

 private:
  GVec b_;
  std::vector<GVec> matrixval_;
  std::vector<GpuBuffer<int>> matrixidx_;

 private:
  void invalidateDevicePtrs() const;
  GpuBuffer<lsReal*>& matrixvalPtrs() const;
  GpuBuffer<int*>& matrixidxPtrs() const;
  mutable GpuBuffer<lsReal*> matrixvalPtrs_;
  mutable GpuBuffer<int*> matrixidxPtrs_;
};
