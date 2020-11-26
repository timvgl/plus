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
                                   const CuParameter conductivity,
                                   const CuParameter pot,
                                   real3 cellsize) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  Grid grid = pot.grid;

  if (!grid.cellInGrid(idx))
    return;

  real vals[7] = {0, 0, 0, 0, 0, 0, 0};
  int colidx[7] = {idx, -1, -1, -1, -1, -1, -1};

  // cell coordinates of the neighbors
  int3 coo = grid.index2coord(idx);
  int3 neighbor[7];
  neighbor[0] = coo + int3{0, 0, 0};
  neighbor[1] = coo + int3{-1, 0, 0};
  neighbor[2] = coo + int3{1, 0, 0};
  neighbor[3] = coo + int3{0, -1, 0};
  neighbor[4] = coo + int3{0, 1, 0};
  neighbor[5] = coo + int3{0, 0, -1};
  neighbor[6] = coo + int3{0, 0, 1};

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
    for (int i = 1; i < sys.nnz; i++) {  // nnz=5 (2D) or nnz=7 (3D)
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

  for (int c = 0; c < sys.nnz; c++) {
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
  bool threeDimenstional = grid.size().z > 1;
  int nNeighbors = threeDimenstional ? 6 : 4;  // nearest neighbors
  int nnz = 1 + nNeighbors;                    // central cell + neighbors
  auto system = std::make_unique<LinearSystem>(grid.ncells(), nnz);
  cudaLaunch(grid.ncells(), k_construct, system->cu(),
             magnet_->conductivity.cu(), magnet_->appliedPotential.cu(),
             magnet_->cellsize());
  return system;
}

__global__ static void k_putSolutionInField(CuField f, lsReal* y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.grid.cellInGrid(idx))
    return;
  f.setValueInCell(idx, 0, (real)y[idx]);
}

Field PoissonSystem::solve() {
  init();
  GVec y = solver_->solve();
  Field pot(magnet_->system(), 1);
  cudaLaunch(pot.grid().ncells(), k_putSolutionInField, pot.cu(), y.get());
  return pot;
}

LinSolver* PoissonSystem::solver() {
  return solver_.get();
}