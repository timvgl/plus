#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

bool demagFieldAssuredZero(const Ferromagnet* magnet) {
  return !magnet->enableDemag;
}

Field evalDemagField(const Ferromagnet* magnet) {
  Field h(magnet->grid(), 3);
  if (demagFieldAssuredZero(magnet)) {
    h.makeZero();
    return h;
  }
  magnet->getMagnetField(magnet)->evalIn(&h);
  return h;
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

Field evalDemagEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->grid(), 1);
  if (demagFieldAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }
  auto h = evalDemagField(magnet);
  cudaLaunch(edens.grid().ncells(), k_demagEnergyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(),
             magnet->msat.cu());
  return edens;
}

real evalDemagEnergy(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return 0.0;

  int ncells = magnet->grid().ncells();
  real edensAverage =
      demagEnergyDensityQuantity(magnet).average()[0];
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}

FM_FieldQuantity demagFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDemagField, 3, "exchange_field", "T");
}

FM_FieldQuantity demagEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDemagEnergyDensity, 1,
                             "demag_energy_density", "J/m3");
}

FM_ScalarQuantity demagEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalDemagEnergy, "demag_energy", "J");
}
