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
    : FerromagnetFieldQuantity(ferromagnet, 3, "demag_field", "T"),
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
  Parameter* msat = &ferromagnet_->msat;
  int ncells = result->grid().ncells();

  convolution_.exec(result, m, msat);

  //// brute method
  // const Field* kernel = demagkernel_.field();
  // cudaLaunch(ncells, k_demagfield, result->cu(), m->cu(), kernel->cu(),
  // msat);
}

DemagEnergyDensity::DemagEnergyDensity(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 1, "demag_energy_density", "J/m3") {
}

__global__ void k_demagEnergyDensity(CuField edens,
                                     CuField hfield,
                                     CuField mag,
                                     CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -0.5 * Ms * dot(m, h));
}

void DemagEnergyDensity::evalIn(Field* result) const {
  auto h = ferromagnet_->demagField()->eval();
  cudaLaunch(result->grid().ncells(), k_demagEnergyDensity, result->cu(),
             ferromagnet_->magnetization()->field()->cu(), h->cu(),
             ferromagnet_->msat.cu());
}

DemagEnergy::DemagEnergy(Ferromagnet* ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "demag_energy", "J") {}

real DemagEnergy::eval() const {
  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = ferromagnet_->demagEnergyDensity()->average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}