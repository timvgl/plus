#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "math.h"
#include "world.hpp"
#include "field.hpp"

DemagField::DemagField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "demag_field", "T"),
      demagkernel_(ferromagnet->grid(),
                   ferromagnet->grid(),
                   ferromagnet->world()->cellsize()) {}

__global__ void k_demagfield(CuField hField,
                             CuField mField,
                             CuField kernel,
                             real msat) {
  if (!hField.cellInGrid())
    return;

  real3 h{0, 0, 0};

  Grid g = mField.grid;
  int3 dstcoo = g.idx2coo(blockIdx.x * blockDim.x + threadIdx.x);

  for (int i = 0; i < g.ncells(); i++) {
    int3 srccoo = g.idx2coo(i);
    int3 dist = dstcoo - srccoo;

    real3 m = mField.cellVector(i);

    real nxx = kernel.cellValue(dist, 0);
    real nyy = kernel.cellValue(dist, 1);
    real nzz = kernel.cellValue(dist, 2);
    real nxy = kernel.cellValue(dist, 3);
    real nxz = kernel.cellValue(dist, 4);
    real nyz = kernel.cellValue(dist, 5);

    h.x -= nxx * m.x + nxy * m.y + nxz * m.z;
    h.y -= nxy * m.x + nyy * m.y + nyz * m.z;
    h.z -= nxz * m.x + nyz * m.y + nzz * m.z;
  }
  const real MU0 = 4 * M_PI * 1e-7;  // TODO: move this to a general place
  hField.setCellVector(msat * MU0 * h);
}

void DemagField::evalIn(Field* result) const {
  const Field* m = ferromagnet_->magnetization()->field();
  const Field* kernel = demagkernel_.field();
  real msat = ferromagnet_->msat;
  int ncells = result->grid().ncells();
  cudaLaunch(ncells, k_demagfield, result->cu(), m->cu(), kernel->cu(), msat);
}