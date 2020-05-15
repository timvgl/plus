#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "world.hpp"
#include "zeeman.hpp"

ExternalField::ExternalField(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "external_field", "T") {}

void ExternalField::evalIn(Field* result) const {
  real3 b_ext = ferromagnet_->world()->biasMagneticField;
  result->setUniformComponent(b_ext.x, 0);
  result->setUniformComponent(b_ext.y, 1);
  result->setUniformComponent(b_ext.z, 2);

  auto magnetFields = ferromagnet_->getMagnetFields();
  for (auto magnetField : magnetFields) {
    // Avoid the demag field, we only want external fields
    if (magnetField->source() == ferromagnet_)
      continue;

    magnetField->addTo(result);
  }
}

bool ExternalField::assuredZero() const {
  auto magnetFields = ferromagnet_->getMagnetFields();
  for (auto magnetField : magnetFields) {
    if (!magnetField->assuredZero()) {
      return false;
    }
  }

  real3 b_ext = ferromagnet_->world()->biasMagneticField;
  return b_ext == real3{0.0, 0.0, 0.0};
}

ZeemanEnergyDensity::ZeemanEnergyDensity(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet,
                               1,
                               "zeeman_energy_density",
                               "J/m3") {}

__global__ void k_zeemanEnergyDensity(CuField edens,
                                      CuField mag,
                                      CuField hfield,
                                      CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -Ms * dot(m, h));
}

void ZeemanEnergyDensity::evalIn(Field* result) const {
  if (assuredZero()) {
    result->makeZero();
    return;
  }

  auto h = ferromagnet_->externalField()->eval();
  cudaLaunch(result->grid().ncells(), k_zeemanEnergyDensity, result->cu(),
             ferromagnet_->magnetization()->field()->cu(), h->cu(),
             ferromagnet_->msat.cu());
}

bool ZeemanEnergyDensity::assuredZero() const {
  return ferromagnet_->externalField()->assuredZero();
}

ZeemanEnergy::ZeemanEnergy(Ferromagnet* ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "zeeman_energy", "J") {}

real ZeemanEnergy::eval() const {
  if (ferromagnet_->zeemanEnergyDensity()->assuredZero())
    return 0.0;

  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = ferromagnet_->zeemanEnergyDensity()->average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}
