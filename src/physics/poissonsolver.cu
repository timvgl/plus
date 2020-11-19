#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "linstepper.hpp"
#include "linsystem.hpp"
#include "poissonsolver.hpp"
#include "reduce.hpp"

PoissonSolver::PoissonSolver(const Ferromagnet* magnet)
    : magnet_(magnet),
      sys_(magnet->grid(), NNEAREST),
      pot_(magnet_->grid(), 1) {
  setMethod(Method::CONJUGATEGRADIENT);
}

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

void PoissonSolver::init() {
  nstep_ = 0;
  cudaLaunch(sys_.grid().ncells(), k_construct, sys_.cu(),
             magnet_->appliedPotential.cu(), magnet_->cellsize());
  stepper_->restart();
}

Field PoissonSolver::solve() {
  init();

  while ((double)residualMaxNorm() > tol) {
    if (nstep_ > maxIterations && maxIterations >= 0) {
      break;
    }

    step();
    nstep_++;
  }

  return pot_;
}

void PoissonSolver::step() {
  stepper_->step();
}

void PoissonSolver::restart() {
  stepper_->restart();
}

void PoissonSolver::setMethod(Method method) {
  stepper_ = LinearSystemSolverStepper::create(&sys_, &pot_, method);
}

void PoissonSolver::setMethodByName(std::string methodName) {
  Method method = LinearSystemSolverStepper::getMethodByName(methodName);
  stepper_ = LinearSystemSolverStepper::create(&sys_, &pot_, method);
}

Field PoissonSolver::state() const {
  return pot_;
}

Field PoissonSolver::residual() const {
  return sys_.residual(pot_);
}

real PoissonSolver::residualMaxNorm() const {
  return maxAbsValue(sys_.residual(pot_));
}