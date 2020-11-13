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

__global__ static void k_apply(CuField y,
                               CuLinearSystem linsys,
                               CuField x,
                               real ka,
                               real kb) {
  int rowidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!y.cellInGrid(rowidx))
    return;

  real ycell = 0.0;

  for (int i = 0; i < linsys.nnz; i++) {
    int colidx = linsys.idx[i][rowidx];
    if (colidx >= 0) {
      ycell +=
          ka * linsys.a[i][rowidx] * x.valueAt(colidx) - kb * linsys.b[rowidx];
    }
  }

  y.setValueInCell(rowidx, 0, ycell);
}

// For a linear system Ax=b, this function returns y= ka * A*x + kb * b
static Field apply(const LinearSystem& sys, const Field& x, real ka, real kb) {
  if (x.ncomp() != 1) {
    throw std::invalid_argument(
        "Applying a linear system matrix multiplication is only possible on a "
        "field argument with 1 component");
  }
  if (x.grid() != sys.grid()) {
    throw std::invalid_argument(
        "The grid of the linear system does not match the grid of the field "
        "argument");
  }
  Field y(sys.grid(), 1);
  cudaLaunch(sys.grid().ncells(), k_apply, y.cu(), sys.cu(), x.cu(), ka, kb);
  return y;
}

Field LinearSystem::matrixmul(const Field& x) const {
  return apply(*this, x, 1.0, 0.0);  // A*x
}

Field LinearSystem::residual(const Field& x) const {
  return apply(*this, x, -1.0, 1.0);  // b - A*x
}

CuLinearSystem LinearSystem::cu() const {
  return CuLinearSystem(grid_, nnz_, matrixidxPtrs_.get(), matrixvalPtrs_.get(),
                        b_.get());
}