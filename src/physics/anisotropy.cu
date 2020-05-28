#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

AnisotropyField::AnisotropyField(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "anisotropy_field", "T") {}

__global__ void k_anisotropyField(CuField hField,
                                  const CuField mField,
                                  CuVectorParameter anisU,
                                  CuParameter Ku1,
                                  CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!hField.cellInGrid(idx))
    return;

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k = Ku1.valueAt(idx);
  real Ms = msat.valueAt(idx);

  real3 h = 2 * k * dot(m, u) * u / Ms;

  hField.setVectorInCell(idx, h);
}

void AnisotropyField::evalIn(Field* result) const {
  if (assuredZero()) {
    result->makeZero();
    return;
  }

  CuField h = result->cu();
  const CuField m = ferromagnet_->magnetization()->field()->cu();
  auto anisU = ferromagnet_->anisU.cu();
  auto ku1 = ferromagnet_->ku1.cu();
  auto msat = ferromagnet_->msat.cu();
  int ncells = ferromagnet_->grid().ncells();
  cudaLaunch(ncells, k_anisotropyField, h, m, anisU, ku1, msat);
}

bool AnisotropyField::assuredZero() const {
  return ferromagnet_->ku1.assuredZero() || ferromagnet_->anisU.assuredZero();
}

AnisotropyEnergyDensity::AnisotropyEnergyDensity(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet,
                               1,
                               "anisotropy_energy_density",
                               "J/m3") {}

__global__ void k_anisotropyEnergyDensity(CuField edens,
                                          CuField mField,
                                          CuVectorParameter anisU,
                                          CuParameter Ku1,
                                          CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  if (msat.valueAt(idx) == 0.0) {
    edens.setValueInCell(idx, 0, 0.0);
    return;
  }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k = Ku1.valueAt(idx);
  edens.setValueInCell(idx, 0, -k * dot(m, u) * dot(m, u));
}

void AnisotropyEnergyDensity::evalIn(Field* edens) const {
  if (assuredZero()) {
    edens->makeZero();
    return;
  }

  CuField e = edens->cu();
  const CuField m = ferromagnet_->magnetization()->field()->cu();
  auto anisU = ferromagnet_->anisU.cu();
  auto ku1 = ferromagnet_->ku1.cu();
  auto msat = ferromagnet_->msat.cu();
  int ncells = ferromagnet_->grid().ncells();
  cudaLaunch(ncells, k_anisotropyEnergyDensity, e, m, anisU, ku1, msat);
}

bool AnisotropyEnergyDensity::assuredZero() const {
  return AnisotropyField(ferromagnet_).assuredZero();
}

AnisotropyEnergy::AnisotropyEnergy(Handle<Ferromagnet> ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "anisotropy_energy", "J") {}

real AnisotropyEnergy::eval() const {
  if (AnisotropyEnergyDensity(ferromagnet_).assuredZero())
    return 0;

  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = AnisotropyEnergyDensity(ferromagnet_).average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}