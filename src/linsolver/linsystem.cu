#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "linsystem.hpp"

LinearSystem::LinearSystem(int nrows, int maxNonZerosInRow)
    : nrows_(nrows),
      nnz_(maxNonZerosInRow),
      matrixval_(maxNonZerosInRow),
      matrixidx_(maxNonZerosInRow) {
  allocate();
}

void LinearSystem::allocate() {
  free();

  b_.allocate(nrows_);

  for (int i = 0; i < nnz_; i++) {
    matrixval_[i].allocate(nrows_);
    matrixidx_[i].allocate(nrows_);
  }

  matrixvalPtrs_.allocate(nnz_);
  matrixidxPtrs_.allocate(nnz_);
  lsReal* h_matrixvalptrs[nnz_];
  int* h_matrixidxptrs[nnz_];
  for (int i = 0; i < nnz_; i++) {
    h_matrixvalptrs[i] = matrixval_[i].get();
    h_matrixidxptrs[i] = matrixidx_[i].get();
  }
  checkCudaError(cudaMemcpyAsync(matrixvalPtrs_.get(), h_matrixvalptrs,
                                 nnz_ * sizeof(lsReal*), cudaMemcpyHostToDevice,
                                 getCudaStream()));
  checkCudaError(cudaMemcpyAsync(matrixidxPtrs_.get(), h_matrixidxptrs,
                                 nnz_ * sizeof(int*), cudaMemcpyHostToDevice,
                                 getCudaStream()));
}

void LinearSystem::free() {
  matrixval_.clear();
  matrixvalPtrs_.recycle();
  matrixidx_.clear();
  matrixidxPtrs_.recycle();
  b_.recycle();
}

__global__ static void k_apply(lsReal* y,
                               CuLinearSystem linsys,
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
  if (x.size() != sys.nrows()) {
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

CuLinearSystem LinearSystem::cu() const {
  return CuLinearSystem(nrows_, nnz_, matrixidxPtrs_.get(),
                        matrixvalPtrs_.get(), b_.get());
}