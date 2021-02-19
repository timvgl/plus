#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "linsystem.hpp"

LinearSystem::LinearSystem() : LinearSystem(0, 0) {}

LinearSystem::LinearSystem(int nrows, int maxNonZerosInRow)
    : b_(nrows),
      matrixval_(maxNonZerosInRow, GVec(nrows)),
      matrixidx_(maxNonZerosInRow, GpuBuffer<int>(nrows)) {}

LinearSystem::LinearSystem(const LinearSystem& other)
    : b_(other.b_),
      matrixval_(other.matrixval_),
      matrixidx_(other.matrixidx_) {}

LinearSystem::LinearSystem(LinearSystem&& other)
    : b_(std::move(other.b_)),
      matrixval_(std::move(other.matrixval_)),
      matrixidx_(std::move(other.matrixidx_)),
      matrixvalPtrs_(std::move(other.matrixvalPtrs_)),
      matrixidxPtrs_(std::move(other.matrixidxPtrs_)) {
  other.clear();
}

LinearSystem& LinearSystem::operator=(const LinearSystem& other) {
  b_ = other.b_;
  matrixval_ = other.matrixval_;
  matrixidx_ = other.matrixidx_;
  invalidateDevicePtrs();
  return *this;
}

LinearSystem& LinearSystem::operator=(LinearSystem&& other) {
  b_ = std::move(other.b_);
  matrixval_ = std::move(other.matrixval_);
  matrixidx_ = std::move(other.matrixidx_);
  matrixvalPtrs_ = std::move(other.matrixvalPtrs_);
  matrixidxPtrs_ = std::move(other.matrixidxPtrs_);
  other.clear();
  return *this;
}

template <class T>
static GpuBuffer<T*> gpuPtrsBuffer(const std::vector<GpuBuffer<T>>& bufs) {
  std::vector<T*> ptrs;
  std::transform(bufs.begin(), bufs.end(), std::back_inserter(ptrs),
                 [](auto& p) { return p.get(); });
  return GpuBuffer<T*>(ptrs);
}

GpuBuffer<lsReal*>& LinearSystem::matrixvalPtrs() const {
  if (matrixvalPtrs_.size() != matrixval_.size())
    matrixvalPtrs_ = gpuPtrsBuffer(matrixval_);
  return matrixvalPtrs_;
}

GpuBuffer<int*>& LinearSystem::matrixidxPtrs() const {
  if (matrixidxPtrs_.size() != matrixidx_.size())
    matrixidxPtrs_ = gpuPtrsBuffer(matrixidx_);
  return matrixidxPtrs_;
}

void LinearSystem::invalidateDevicePtrs() const {
  matrixidxPtrs_.recycle();
  matrixvalPtrs_.recycle();
}

void LinearSystem::clear() {
  matrixval_.clear();
  matrixidx_.clear();
  b_.recycle();
  invalidateDevicePtrs();
}

__global__ static void k_apply(lsReal* y,
                               const LinearSystem::CuData linsys,
                               lsReal* x,
                               lsReal ka,
                               lsReal kb) {
  int rowidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowidx >= linsys.nrows)
    return;

  lsReal ycell = 0.0;

  for (int i = 0; i < linsys.nnz; i++) {
    int colidx = linsys.idx[i][rowidx];
    if (colidx >= 0) {
      ycell += ka * linsys.a[i][rowidx] * x[colidx] - kb * linsys.b[rowidx];
    }
  }

  y[rowidx] = ycell;
}

// For a linear system Ax=b, this function returns y= ka * A*x + kb * b
static GVec apply(const LinearSystem& sys,
                  const GVec& x,
                  lsReal ka,
                  lsReal kb) {
  if (x.size() != sys.nRows()) {
    throw std::invalid_argument(
        "The numbers of rows in the linear system does not match the number of "
        "cells of the field");
  }
  GVec y(x.size());
  cudaLaunch(x.size(), k_apply, y.get(), sys.cu(), x.get(), ka, kb);
  return y;
}

GVec LinearSystem::matrixmul(const GVec& x) const {
  return apply(*this, x, 1.0, 0.0);  // A*x
}

GVec LinearSystem::residual(const GVec& x) const {
  return apply(*this, x, -1.0, 1.0);  // b - A*x
}

LinearSystem::CuData LinearSystem::cu() const {
  return CuData{nRows(), nNonZerosInMatrixRow(), matrixidxPtrs().get(),
                matrixvalPtrs().get(), b_.get()};
}
