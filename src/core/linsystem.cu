#include <stdexcept>

#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "linsystem.hpp"

LinearSystem::LinearSystem(Grid grid, int maxNonZerosInRow)
    : grid_(grid),
      nnz_(maxNonZerosInRow),
      matrixval_(maxNonZerosInRow),
      matrixidx_(maxNonZerosInRow) {
  allocate();
}

void LinearSystem::allocate() {
  free();

  b_.allocate(grid_.ncells());

  for (int i = 0; i < nnz_; i++) {
    matrixval_[i].allocate(grid_.ncells());
    matrixidx_[i].allocate(grid_.ncells());
  }

  matrixvalPtrs_.allocate(nnz_);
  matrixidxPtrs_.allocate(nnz_);
  real* h_matrixvalptrs[nnz_];
  int* h_matrixidxptrs[nnz_];
  for (int i = 0; i < nnz_; i++) {
    h_matrixvalptrs[i] = matrixval_[i].get();
    h_matrixidxptrs[i] = matrixidx_[i].get();
  }
  checkCudaError(cudaMemcpyAsync(matrixvalPtrs_.get(), h_matrixvalptrs,
                                 nnz_ * sizeof(real*), cudaMemcpyHostToDevice,
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

__global__ static void k_matrixmul(CuField y,
                                   CuLinearSystem linsys,
                                   CuField x) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!y.cellInGrid(idx))
    return;

  real ycell = 0.0;

  for (int i = 0; i < linsys.nnz; i++) {
    ycell += linsys.a[i][idx] * x.valueAt(idx);
  }

  y.setValueInCell(idx, 0, ycell);
}

Field LinearSystem::matrixmul(const Field& x) const {
  if (x.ncomp() != 1) {
    throw std::invalid_argument(
        "Applying a linear system matrix multiplication is only possible on a "
        "field argument with 1 component");
  }
  if (x.grid() != grid_) {
    throw std::invalid_argument(
        "The grid of the linear system does not match the grid of the field "
        "argument");
  }

  Field y = Field(grid_,1);
  cudaLaunch(grid_.ncells(), k_matrixmul, y.cu(), cu(), x.cu());
  return y;
}

CuLinearSystem LinearSystem::cu() const {
  return CuLinearSystem(grid_, nnz_, matrixidxPtrs_.get(), matrixvalPtrs_.get(),
                        b_.get());
}