#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"
#include "poissonsystem.hpp"
#include "reduce.hpp"

PoissonSystem::PoissonSystem(const Ferromagnet* magnet) : magnet_(magnet) {}

__global__ static void k_construct(CuLinearSystem sys,
                                   const CuField conductivity,
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

  // conductivity at the center cell
  real c0 = conductivity.valueAt(coo);

  // if potential applied, set potential directly
  if (!isnan(pot.valueAt(idx))) {
    vals[0] = 1.0;
    sys.b[idx] = pot.valueAt(idx);

    // set potential to 0 if conductivity is zero
  } else if (c0 == real(0.0)) {
    vals[0] = 1.0;
    sys.b[idx] = 0.0;

  } else {
    for (int i = 1; i < 5; i++) {
      if (grid.cellInGrid(neighbor[i])) {
        real c_ = conductivity.valueAt(neighbor[i]);
        real c = sqrt(c_ * c0);  // geometric mean
        vals[0] += c;
        vals[i] -= c;
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

void PoissonSystem::init() {
  solver_ = std::make_unique<LinSolver>(construct());
  solver_->restartStepper();
}

std::unique_ptr<LinearSystem> PoissonSystem::construct() const {
  Grid grid = magnet_->grid();
  auto system = std::make_unique<LinearSystem>(grid, NNEAREST);
  Field conductivity = magnet_->conductivity.eval();
  cudaLaunch(grid.ncells(), k_construct, system->cu(), conductivity.cu(),
             magnet_->appliedPotential.cu(), magnet_->cellsize());
  return system;
}

Field PoissonSystem::solve() {
  init();
  return solver_->solve();
}

LinSolver* PoissonSystem::solver() {
  return solver_.get();
}