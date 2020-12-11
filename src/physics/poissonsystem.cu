#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linsolver.hpp"
#include "linsystem.hpp"
#include "poissonsystem.hpp"
#include "reduce.hpp"

/** Returns the neighbor index for a given relative position.
 *  This index correspond to the index of this position in the neighbors array.
 */
__device__ static constexpr int neighbor_id(int3 rcoo) {
  int id = 0;
  id += 1 * (rcoo.x == -1 ? 2 : rcoo.x);
  id += 3 * (rcoo.y == -1 ? 2 : rcoo.y);
  id += 9 * (rcoo.z == -1 ? 2 : rcoo.z);
  return id;
}

/** List of the relative positions of the neighbors */
__constant__ const int3 neighbors[27] = {
    // clang-format off
    int3{  0,  0,  0}, // id = 0     output of neighbor_id
    int3{  1,  0,  0}, // id = 1
    int3{ -1,  0,  0}, // id = 2
    int3{  0,  1,  0}, // id = 3
    int3{  1,  1,  0}, // id = 4
    int3{ -1,  1,  0}, // id = 5
    int3{  0, -1,  0}, // id = 6
    int3{  1, -1,  0}, // id = 7
    int3{ -1, -1,  0}, // id = 8
    int3{  0,  0,  1}, // id = 9     3D from here on
    int3{  1,  0,  1}, // id = 10
    int3{ -1,  0,  1}, // id = 11
    int3{  0,  1,  1}, // id = 12
    int3{  1,  1,  1}, // id = 13
    int3{ -1,  1,  1}, // id = 14
    int3{  0, -1,  1}, // id = 15
    int3{  1, -1,  1}, // id = 16
    int3{ -1, -1,  1}, // id = 17
    int3{  0,  0, -1}, // id = 18
    int3{  1,  0, -1}, // id = 19
    int3{ -1,  0, -1}, // id = 20
    int3{  0,  1, -1}, // id = 21
    int3{  1,  1, -1}, // id = 22
    int3{ -1,  1, -1}, // id = 23
    int3{  0, -1, -1}, // id = 24
    int3{  1, -1, -1}, // id = 25
    int3{ -1, -1, -1}, // id = 26
    // clang-format on
};

/** Represent a sparce matrix row with max N non zero elements. */
class Row {
 public:
  /** Maximal non zero values in a sparse row.
   *  1 center cell + 6 nearest neighbors + 20 next nearest neighbors
   */
  static const int N = 27;

  real b; /** Right hand sight of this row in Ax=b */

 private:
  int rowidx;     /** Row index of this sparse row  */
  real value_[N]; /** matrix values in this sparse row */
  int colidx_[N]; /** column indices of the matrix values in this sparse row */

 public:
  __device__ Row(int rowidx, CuSystem system) : b(0), rowidx(rowidx) {
    const int3 coo = system.grid.index2coord(rowidx);
    for (auto rcoo : neighbors) {
      value(rcoo) = 0.0;
      if (system.inGeometry(coo + rcoo)) {
        colidx(rcoo) = system.grid.coord2index(coo + rcoo);
      } else {
        colidx(rcoo) = -1;
      }
    }
  }

  /** Return Column index for cell with relative coordinates rcoo to center
   *  cell. If cell is outside the geometry, return -1.
   */
  __device__ int& colidx(int3 rcoo) { return colidx_[neighbor_id(rcoo)]; }

  /** Matrix value for cell with relative coordinate rcoo to center cell. */
  __device__ real& value(int3 rcoo) { return value_[neighbor_id(rcoo)]; }

  /** Return true if cell with relative coordinate rcoo to center cell is inside
   * the geometry. */
  __device__ bool inGeometry(int3 rcoo) { return colidx(rcoo) < 0; }

  /** Add a finite difference in the matrix row. */
  __device__ void addDiff(int3 rcoo1, int3 rcoo2, real val) {
    int nid1 = neighbor_id(rcoo1);
    int nid2 = neighbor_id(rcoo2);
    if (colidx_[nid1] >= 0 && colidx_[nid2] >= 0) {
      value_[nid1] += val;
      value_[nid2] -= val;
    }
  }

  /** Fill in the row in the linear system linsys. */
  __device__ void writeRowInLinearSystem(CuLinearSystem* linsys) {
    // Set the right hand side of the row in the system of linear equations
    linsys->b[rowidx] = b;

    // Fill in the non zero matrix values
    int c = 0;
    for (int3 rcoo : neighbors) {
      if (value(rcoo) != 0.0 && colidx(rcoo) >= 0) {  // element is non zero
        linsys->idx[c][rowidx] = colidx(rcoo);
        linsys->a[c][rowidx] = value(rcoo) / value({0, 0, 0});
        c++;
      }
    }

    // Invalidate other available places by setting an invalid column index.
    for (; c < linsys->nnz; c++)
      linsys->idx[c][rowidx] = -1;
  }
};

PoissonSystem::PoissonSystem(const Ferromagnet* magnet) : magnet_(magnet) {}

__global__ static void k_construct(CuLinearSystem linsys,
                                   const CuParameter conductivity,
                                   const CuParameter pot) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  CuSystem system = pot.system;

  // if outside the grid, then there is nothing to do, and we can return early
  if (!system.grid.cellInGrid(idx))
    return;

  Row row(idx, system);

  if (!system.inGeometry(idx) || conductivity.valueAt(idx) == real(0.0) ||
      !isnan(pot.valueAt(idx))) {
    // Put 1 on the diagonal
    row.colidx({0, 0, 0}) = idx;
    row.value({0, 0, 0}) = 1.0;
    // set applied potential as rhs (or 0 if there is no applied potential)
    row.b = isnan(pot.valueAt(idx)) ? 0.0 : pot.valueAt(idx);

  } else {  // NON TRIVIAL MATRIX ROW

    // Return the average conductivity of two cells
    auto avgConductivity = [&conductivity](int idx1, int idx2) {
      real c1 = conductivity.valueAt(idx1);
      real c2 = conductivity.valueAt(idx2);
      return sqrt(c1 * c2);
    };

    const real dx = system.cellsize.x;
    const real dy = system.cellsize.y;
    const real dz = system.cellsize.z;

    // current along x direction
    for (int ix : {-1, 1}) {
      if (!row.inGeometry({ix, 0, 0})) {
        int idx_ = row.colidx({ix, 0, 0});
        real fac = dy * dz * avgConductivity(idx, idx_);
        row.addDiff({0, 0, 0}, {ix, 0, 0}, fac / dx);
      }
    }

    // current along y direction
    for (int iy : {-1, 1}) {
      if (!row.inGeometry({0, iy, 0})) {
        int idx_ = row.colidx({0, iy, 0});
        real fac = dx * dz * avgConductivity(idx, idx_) / dy;
        row.addDiff({0, 0, 0}, {0, iy, 0}, fac);
      }
    }

    // current along z direction
    for (int iz : {-1, 1}) {
      if (!row.inGeometry({0, 0, iz})) {
        int idx_ = row.colidx({0, 0, iz});
        real fac = dx * dy * avgConductivity(idx, idx_) / dz;
        row.addDiff({0, 0, 0}, {0, 0, iz}, fac);
      }
    }

    row.b = 0.0;
  }

  row.writeRowInLinearSystem(&linsys);
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
