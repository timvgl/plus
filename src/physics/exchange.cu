#include "cudalaunch.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "handler.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"

bool exchangeAssuredZero(const Ferromagnet* magnet) {
  return magnet->aex.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__global__ void k_exchangeField(CuField hField,
                                CuField mField,
                                CuParameter aex,
                                CuParameter msat,
                                real3 cellsize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!hField.cellInGrid(idx))
    return;

  int3 coo = hField.grid.index2coord(idx);

  real3 m = mField.vectorAt(idx);
  real a = aex.valueAt(idx) / msat.valueAt(idx);

  real3 h{0, 0, 0};  // accumulate exchange field in cell idx

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

  for (int3 relcoo : neighborRelativeCoordinates) {
    int3 coo_ = coo + relcoo;
    int idx_ = hField.grid.coord2index(coo_);

    if (hField.cellInGrid(coo_)) {
      real dr =
          cellsize.x * relcoo.x + cellsize.y * relcoo.y + cellsize.z * relcoo.z;
      real3 m_ = mField.vectorAt(idx_);
      real a_ = aex.valueAt(idx_) / msat.valueAt(idx_);

      h += 2 * harmonicMean(a, a_) * (m_ - m) / (dr * dr);
    }
  }

  hField.setVectorInCell(idx, h);
}

Field evalExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->grid(), 3);
  if (exchangeAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  cudaLaunch(hField.grid().ncells(), k_exchangeField, hField.cu(),
             magnet->magnetization()->field()->cu(), magnet->aex.cu(),
             magnet->msat.cu(), magnet->world()->cellsize());
  return hField;
}

__global__ void k_exchangeEnergyDensity(CuField edens,
                                        CuField mag,
                                        CuField hfield,
                                        CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -0.5 * Ms * dot(m, h));
}

Field evalExchangeEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->grid(), 1);
  if (exchangeAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }
  Field h = evalExchangeField(magnet);
  cudaLaunch(edens.grid().ncells(), k_exchangeEnergyDensity, edens.cu(),
             magnet->magnetization()->field()->cu(), h.cu(), magnet->msat.cu());
  return edens;
}

real evalExchangeEnergy(const Ferromagnet* magnet) {
  if (exchangeAssuredZero(magnet))
    return 0;
  real edens = exchangeEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExchangeField, 3, "exchange_field",
                             "T");
}

FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExchangeEnergyDensity, 1,
                             "exchange_energy_density", "J/m3");
}

FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalExchangeEnergy, "exchange_energy",
                              "J");
}
