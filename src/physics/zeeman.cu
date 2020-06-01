#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "ferromagnet.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "world.hpp"
#include "zeeman.hpp"

bool externalFieldAssuredZero(const Ferromagnet* magnet) {
  auto magnetFields = magnet->getMagnetFields();
  for (auto magnetField : magnetFields) {
    if (!magnetField->assuredZero()) {
      return false;
    }
  }

  real3 b_ext = magnet->world()->biasMagneticField;
  return b_ext == real3{0.0, 0.0, 0.0};
}

Field evalExternalField(const Ferromagnet* magnet) {
  Field h(magnet->grid(), 3);
  if (externalFieldAssuredZero(magnet)) {
    h.makeZero();
    return h;
  }

  real3 b_ext = magnet->world()->biasMagneticField;
  h.setUniformComponent(b_ext.x, 0);
  h.setUniformComponent(b_ext.y, 1);
  h.setUniformComponent(b_ext.z, 2);

  auto magnetFields = magnet->getMagnetFields();
  for (auto magnetField : magnetFields) {
    // Avoid the demag field, we only want external fields
    if (magnetField->source() == magnet)
      continue;

    magnetField->addToField(&h);
  }
  return h;
}


__global__ void k_zeemanEnergyDensity(CuField edens,
                                      const CuField mag,
                                      const CuField hfield,
                                      const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -Ms * dot(m, h));
}

Field evalZeemanEnergyDensity(const Ferromagnet* magnet){
  Field edens(magnet->grid(),1);
  if (externalFieldAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }
  Field h = evalExternalField(magnet);
  cudaLaunch(edens.grid().ncells(), k_zeemanEnergyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(),
             magnet->msat.cu());
  return edens;
}

real zeemanEnergy(const Ferromagnet* magnet){
  if (externalFieldAssuredZero(magnet)) {
    return 0.0;
  }

  real edens = zeemanEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity externalFieldQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalExternalField, 3, "external_field", "T");
}

FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalZeemanEnergyDensity, 1,
                             "zeeman_energy_density", "J/m3");
}

FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet * magnet) {
  return FM_ScalarQuantity(magnet, zeemanEnergy, "zeeman_energy", "J");
}