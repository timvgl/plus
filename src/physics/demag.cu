#include "constants.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "demagconvolution.hpp"
#include "demagkernel.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "math.h"
#include "world.hpp"

DemagField::DemagField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "demag_field", "T"),
      convolution_(ferromagnet->grid(), ferromagnet->world()->cellsize()),
      demagkernel_(ferromagnet->grid(),
                   ferromagnet->grid(),
                   ferromagnet->world()->cellsize()) {}

__global__ void k_demagfield(CuField hField,
                             CuField mField,
                             CuField kernel,
                             real msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGrid(idx))
    return;

  real3 h{0, 0, 0};

  Grid g = mField.grid;
  int3 dstcoo = g.index2coord(idx);

  for (int i = 0; i < g.ncells(); i++) {
    int3 srccoo = g.index2coord(i);
    int3 dist = dstcoo - srccoo;

    real3 m = mField.vectorAt(i);

    real nxx = kernel.valueAt(dist, 0);
    real nyy = kernel.valueAt(dist, 1);
    real nzz = kernel.valueAt(dist, 2);
    real nxy = kernel.valueAt(dist, 3);
    real nxz = kernel.valueAt(dist, 4);
    real nyz = kernel.valueAt(dist, 5);

    h.x -= nxx * m.x + nxy * m.y + nxz * m.z;
    h.y -= nxy * m.x + nyy * m.y + nyz * m.z;
    h.z -= nxz * m.x + nyz * m.y + nzz * m.z;
  }

  hField.setVectorInCell(idx, msat * MU0 * h);
}

void DemagField::evalIn(Field* result) const {
  const Field* m = ferromagnet_->magnetization()->field();
  const Field* kernel = demagkernel_.field();
  real msat = ferromagnet_->msat;
  int ncells = result->grid().ncells();

  convolution_.exec(result, m, msat);

  //// brute method
  // cudaLaunch(ncells, k_demagfield, result->cu(), m->cu(), kernel->cu(),
  // msat);
}