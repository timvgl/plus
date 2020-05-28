#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

DemagField::DemagField(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "demag_field", "T")
 {}

void DemagField::evalIn(Field* result) const {
  if (assuredZero()) {
    result->makeZero();
    return;
  }
  ferromagnet_->getMagnetField(ferromagnet_)->evalIn(result);
}

bool DemagField::assuredZero() const {
  return !ferromagnet_->enableDemag;
}

DemagEnergyDensity::DemagEnergyDensity(Handle<Ferromagnet> ferromagnet)
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
  if (assuredZero()) {
    result->makeZero();
    return;
  }
  auto h = DemagField(ferromagnet_).eval();
  cudaLaunch(result->grid().ncells(), k_demagEnergyDensity, result->cu(),
             ferromagnet_->magnetization()->field()->cu(), h->cu(),
             ferromagnet_->msat.cu());
}

bool DemagEnergyDensity::assuredZero() const {
  return DemagField(ferromagnet_).assuredZero();
}

DemagEnergy::DemagEnergy(Handle<Ferromagnet> ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "demag_energy", "J") {}

real DemagEnergy::eval() const {
  if (DemagEnergyDensity(ferromagnet_).assuredZero())
    return 0.0;

  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = DemagEnergyDensity(ferromagnet_).average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}