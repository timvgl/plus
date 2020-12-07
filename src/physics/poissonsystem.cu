#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"
#include "poissonsystem.hpp"
#include "reduce.hpp"

/** Represent a sparce matrix row with max N non zero elements. */
template <int N>
struct Row {
  real value[N] = {0.0};
  int colidx[N] = {-1};  // initialize with invalid indices

  __device__ Row(int idx) { colidx[0] = idx; }
  __device__ void addDiff(int lid1, int lid2, real val) {
    value[lid1] += val;
    value[lid2] -= val;
  }
};

//* Returns a local id for the central cell and its nearest neighbors. */
__device__ constexpr int lid_nearest(int ix, int iy, int iz) {
  // clang-format off
  if      (ix ==  0 && iy ==  0 && iz ==  0) return 0; // id for central cell
  else if (ix == -1 && iy ==  0 && iz ==  0) return 1; // id for left cell
  else if (ix ==  1 && iy ==  0 && iz ==  0) return 2; // ...
  else if (ix ==  0 && iy == -1 && iz ==  0) return 3;
  else if (ix ==  0 && iy ==  1 && iz ==  0) return 4;
  // 3D
  else if (ix ==  0 && iy ==  0 && iz == -1) return 5;
  else if (ix ==  0 && iy ==  0 && iz ==  1) return 6;
  return -1;
  // clang-format on
}

PoissonSystem::PoissonSystem(const Ferromagnet* magnet) : magnet_(magnet) {}

__global__ static void k_construct(CuLinearSystem linsys,
                                   const CuParameter conductivity,
                                   const CuParameter pot) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  CuSystem system = pot.system;

  if (!system.grid.cellInGrid(idx))
    return;

  // short alias, since we will use this a lot
  constexpr auto lid = lid_nearest;

  // coordinate of central cell
  int3 coo = system.grid.index2coord(idx);

  // Row values for row idx
  Row<7> row(idx);

  // if potential applied, set potential directly
  if (!isnan(pot.valueAt(idx))) {
    row.value[0] = 1.0;
    linsys.b[idx] = pot.valueAt(idx);

    // set potential to 0 if outside geometry or conductivity is zero
  } else if (!system.inGeometry(idx) ||
             conductivity.valueAt(idx) == real(0.0)) {
    row.value[0] = 1.0;
    linsys.b[idx] = 0.0;

    // Compute the current flowing from the center cell to the neighboring cells
  } else {
    // Return the average conductivity of two cells
    auto avgConductivity = [&conductivity](int idx1, int idx2) {
      real c1 = conductivity.valueAt(idx1);
      real c2 = conductivity.valueAt(idx2);
      return sqrt(c1 * c2);
    };

    real dx = system.cellsize.x;
    real dy = system.cellsize.y;
    real dz = system.cellsize.z;

    // current along x direction
    for (int ix : {-1, 1}) {
      if (system.inGeometry(coo + int3{ix, 0, 0})) {
        int idx_ = system.grid.coord2index(coo + int3{ix, 0, 0});
        row.colidx[lid(ix, 0, 0)] = idx_;
        real fac = dy * dz * avgConductivity(idx, idx_) / dx;
        row.addDiff(lid(0, 0, 0), lid(ix, 0, 0), fac);
      }
    }

    // current along y direction
    for (int iy : {-1, 1}) {
      int3 coo_ = coo + int3{0, iy, 0};
      if (system.inGeometry(coo_)) {
        int idx_ = system.grid.coord2index(coo_);
        row.colidx[lid(0, iy, 0)] = idx_;
        real fac = dx * dz * avgConductivity(idx, idx_) / dy;
        row.addDiff(lid(0, 0, 0), lid(0, iy, 0), fac);
      }
    }

    // current along z direction
    for (int iz : {-1, 1}) {
      int3 coo_ = coo + int3{0, 0, iz};
      if (system.inGeometry(coo_)) {
        int idx_ = system.grid.coord2index(coo_);
        row.colidx[lid(0, 0, iz)] = idx_;
        real fac = dx * dy * avgConductivity(idx, idx_) / dz;
        row.addDiff(lid(0, 0, 0), lid(0, 0, iz), fac);
      }
    }

    linsys.b[idx] = 0.0;
  }

  for (int c = 0; c < linsys.nnz; c++) {
    linsys.idx[c][idx] = row.colidx[c];
    linsys.a[c][idx] = row.value[c] / row.value[0];
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
             magnet_->conductivity.cu(), magnet_->appliedPotential.cu());
  return system;
}

__global__ static void k_putSolutionInField(CuField f, lsReal* y) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!f.cellInGeometry(idx))
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
