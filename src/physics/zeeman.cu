#include "antiferromagnet.hpp"
#include "antiferromagnetquantity.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnet.hpp"
#include "mumaxworld.hpp"
#include "zeeman.hpp"

bool externalFieldAssuredZero(const Magnet* magnet) {
  auto strayFields = magnet->getStrayFields();
  for (auto strayField : strayFields) {
    if (!strayField->assuredZero()) {
      return false;
    }
  }

  const MumaxWorld* world = static_cast<const MumaxWorld*>(magnet->world());
  real3 b_ext = world->biasMagneticField;
  return b_ext == real3{0.0, 0.0, 0.0};
}

Field calcExternalFields(const Magnet* magnet, Field h, const FM_VectorParameter& mbias_field) {

  if (externalFieldAssuredZero(magnet)) {
    h.makeZero();
    return h;
  }

  const MumaxWorld* world = static_cast<const MumaxWorld*>(magnet->world());
  real3 wB_bias = world->biasMagneticField; // bias field on world

  h.setUniformComponent(0, wB_bias.x);
  h.setUniformComponent(1, wB_bias.y);
  h.setUniformComponent(2, wB_bias.z);

  mbias_field.addToField(h);
    
  auto strayFields = magnet->getStrayFields();
  for (auto strayField : strayFields) {
    // Avoid the demag field, we only want external fields
    if (strayField->source() == magnet)
      continue;
    strayField->addToField(h);
  }
  return h;
}

Field evalExternalField(const Ferromagnet* magnet) {
  Field h(magnet->system(), 3);
  auto& mB_bias = magnet->biasMagneticField; // bias field on individual magnet
  return calcExternalFields(magnet, h, mB_bias);
}

Field evalAFMExternalField(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  Field h(sublattice->system(), 3);
  auto& mB_bias = sublattice->biasMagneticField; // bias field on individual magnet
  return calcExternalFields(magnet, h, mB_bias);
}

Field evalZeemanEnergyDensity(const Ferromagnet* magnet) {
  if (externalFieldAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalExternalField(magnet), 1.0);
}

real zeemanEnergy(const Ferromagnet* magnet) {
  if (externalFieldAssuredZero(magnet))
    return 0.0;

  real edens = zeemanEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity externalFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExternalField, 3, "external_field", "T");
}

AFM_FieldQuantity AFM_externalFieldQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  return AFM_FieldQuantity(magnet, sublattice, evalAFMExternalField, 3, "external_field", "T");
}

FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalZeemanEnergyDensity, 1, "zeeman_energy_density", "J/m3");
}

FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, zeemanEnergy, "zeeman_energy", "J");
}