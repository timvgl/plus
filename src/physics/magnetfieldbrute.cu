#include "cudalaunch.hpp"
#include "field.hpp"
#include "magnetfieldbrute.hpp"
#include "parameter.hpp"

// TODO: figure out why including constants.hpp leads to errors
__device__ static const real PI = 3.14159265358979323846;
__device__ static const real MU0 = 4 * PI * 1e-7;

__global__ void k_demagfield(CuField hField,
                             CuField mField,
                             CuField kernel,
                             CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGrid(idx))
    return;

  real3 h{0, 0, 0};

  int3 dstcoo = hField.grid.index2coord(idx);

  for (int i = 0; i < mField.grid.ncells(); i++) {
    int3 srccoo = mField.grid.index2coord(i);
    int3 dist = srccoo - dstcoo;

    real3 M = msat.valueAt(i) * mField.vectorAt(i);

    real nxx = kernel.valueAt(dist, 0);
    real nyy = kernel.valueAt(dist, 1);
    real nzz = kernel.valueAt(dist, 2);
    real nxy = kernel.valueAt(dist, 3);
    real nxz = kernel.valueAt(dist, 4);
    real nyz = kernel.valueAt(dist, 5);

    h.x -= nxx * M.x + nxy * M.y + nxz * M.z;
    h.y -= nxy * M.x + nyy * M.y + nyz * M.z;
    h.z -= nxz * M.x + nyz * M.y + nzz * M.z;
  }

  hField.setVectorInCell(idx, MU0 * h);
}

MagnetFieldBruteExecutor::MagnetFieldBruteExecutor(Grid gridOut,
                                                   Grid gridIn,
                                                   real3 cellsize)
    : kernel_(gridOut, gridIn, cellsize) {}

void MagnetFieldBruteExecutor::exec(Field* h,
                                    const Field* m,
                                    const Parameter* msat) const {
  // TODO: check dimensions of fields
  const Field* kernel = kernel_.field();
  int ncells = h->grid().ncells();
  cudaLaunch(ncells, k_demagfield, h->cu(), m->cu(), kernel->cu(), msat->cu());
}