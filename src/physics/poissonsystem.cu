#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "poissonsystem.hpp"
#include "stdint.h"

__global__ static void k_construct(CuField matrix,
                                   CuField rhs,
                                   const CuParameter pot,
                                   real3 cellsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  Grid grid = matrix.grid;

  if (!grid.cellInGrid(idx))
    return;

  real vals[5] = {0, 0, 0, 0, 0};

  if (!isnan(pot.valueAt(idx))) {
    vals[0] = 1.0;
    rhs.setValueInCell(idx, 0, pot.valueAt(idx));
  } else {
    int3 coo = grid.index2coord(idx);
    int3 relcoo[5] = {int3{0, 0, 0}, int3{-1, 0, 0}, int3{1, 0, 0},
                      int3{0, -1, 0}, int3{0, 1, 0}};
    for (int i = 1; i < 5; i++) {
      if (grid.cellInGrid(coo + relcoo[i])) {
        vals[0] += 1.0;
        vals[i] -= 1.0;
      }
    }
    rhs.setValueInCell(idx, 0, 0.0);
  }

  for (int c = 0; c < 5; c++) {
    matrix.setValueInCell(idx, c, vals[c] / vals[0]);
  }
}

void PoissonSystem::construct() {
  matrixValues_ = Field(grid(), 5);
  rhs_ = Field(grid(), 1);
  cudaLaunch(matrixValues_.grid().ncells(), k_construct, matrixValues_.cu(),
             rhs_.cu(), magnet_->appliedPotential.cu(), magnet_->cellsize());
}

__global__ static void k_apply(CuField yField, CuField A, CuField xField) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  Grid grid = yField.grid;

  if (!grid.cellInGrid(idx))
    return;

  int3 coo = grid.index2coord(idx);
  int3 neighbors[5];
  neighbors[0] = coo + int3{0, 0, 0};
  neighbors[1] = coo + int3{-1, 0, 0};
  neighbors[2] = coo + int3{1, 0, 0};
  neighbors[3] = coo + int3{0, -1, 0};
  neighbors[4] = coo + int3{0, 1, 0};

  real y = 0.0;

#pragma unroll
  for (int i = 0; i < 5; i++) {
    int3 coo_ = neighbors[i];
    if (grid.cellInGrid(coo_))
      y += A.valueAt(idx, i) * xField.valueAt(coo_);
  }

  yField.setValueInCell(idx, 0, y);
}

Field PoissonSystem::apply(const Field& x) {
  if (x.grid() != grid())
    throw std::invalid_argument(
        "Grid of the field argument does not match the grid of the system "
        "matrix");

  Field y = Field(grid(), 1);
  cudaLaunch(matrixValues_.grid().ncells(), k_apply, y.cu(), matrixValues_.cu(),
             x.cu());
  return y;
}

Field PoissonSystem::solve() {
  construct();
  Field x = Field(grid(), 1, 0.0);
  int nstep = 1000;
  for (int i = 0; i < nstep; i++) {
    Field Ax = apply(x);                 // Ax = A*x
    Field r = add(1.0, Ax, -1.0, rhs_);  // r = Ax-b
    x = add(1.0, x, -1.0, r);            // x = x-r
  }
  return x;
}