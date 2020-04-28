#include "cudalaunch.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

ExchangeField::ExchangeField(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "exchange_field", "T") {}

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

void exchangeField(Field* hField, const Ferromagnet* ferromagnet) {
  cudaLaunch(hField->grid().ncells(), k_exchangeField, hField->cu(),
             ferromagnet->magnetization()->field()->cu(), ferromagnet->aex.cu(),
             ferromagnet->msat.cu(), ferromagnet->world()->cellsize());
}

void ExchangeField::evalIn(Field* result) const {
  exchangeField(result, ferromagnet_);
}

ExchangeEnergyDensity::ExchangeEnergyDensity(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet,
                               1,
                               "exchange_energy_density",
                               "J/m3") {}

__global__ void k_exchangeEnergyDensity(CuField edens,
                                        const CuField mag,
                                        const CuField hfield,
                                        const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -0.5 * Ms * dot(m, h));
}

void ExchangeEnergyDensity::evalIn(Field* result) const {
  auto h = ferromagnet_->exchangeField()->eval();
  cudaLaunch(result->grid().ncells(), k_exchangeEnergyDensity, result->cu(),
             ferromagnet_->magnetization()->field()->cu(), h->cu(),
             ferromagnet_->msat.cu());
}
