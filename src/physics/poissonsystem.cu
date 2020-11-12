#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linsystem.hpp"
#include "poissonsystem.hpp"
#include "stdint.h"

__global__ static void k_construct(CuLinearSystem sys,
                                   const CuParameter pot,
                                   real3 cellsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  Grid grid = sys.grid;

  if (!grid.cellInGrid(idx))
    return;

  real vals[5] = {0, 0, 0, 0, 0};
  int colidx[5] = {idx, -1, -1, -1, -1};

  // cell coordinates of the neighbors
  int3 coo = grid.index2coord(idx);
  int3 neighbor[5];
  neighbor[0] = coo + int3{0, 0, 0};
  neighbor[1] = coo + int3{-1, 0, 0};
  neighbor[2] = coo + int3{1, 0, 0};
  neighbor[3] = coo + int3{0, -1, 0};
  neighbor[4] = coo + int3{0, 1, 0};

  if (!isnan(pot.valueAt(idx))) {
    vals[0] = 1.0;
    sys.b[idx] = pot.valueAt(idx);
  } else {
    for (int i = 1; i < 5; i++) {
      if (grid.cellInGrid(neighbor[i])) {
        vals[0] += 1.0;
        vals[i] -= 1.0;
        colidx[i] = grid.coord2index(neighbor[i]);
      }
    }
    sys.b[idx] = 0.0;
  }

  for (int c = 0; c < 5; c++) {
    sys.idx[c][idx] = colidx[c];
    sys.a[c][idx] = vals[c] / vals[0];
  }
}

void PoissonSystem::construct() {
  cudaLaunch(sys_.grid().ncells(), k_construct, sys_.cu(),
             magnet_->appliedPotential.cu(), magnet_->cellsize());
}

Field PoissonSystem::solve() {
  construct();
  Field x = Field(grid(), 1, 0.0);
  int nstep = 1000;
  for (int i = 0; i < nstep; i++) {
    Field r = sys_.residual(x);  // r = Ax-b
    x = add(1.0, x, -1.0, r);    // x = x-r
  }
  return x;
}