#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "mumaxworld.hpp"
#include "zeeman.hpp"

bool externalFieldAssuredZero(const Ferromagnet* magnet) {
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

Field evalExternalField(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();

  Field h(magnet->system(), comp);
  if (externalFieldAssuredZero(magnet)) {
    h.makeZero();
    return h;
  }
  const MumaxWorld* world = static_cast<const MumaxWorld*>(magnet->world());
  real3 wB_bias = world->biasMagneticField;
  auto& mB_bias = magnet->biasMagneticField;

  h.setUniformComponent(0, wB_bias.x);
  h.setUniformComponent(1, wB_bias.y);
  h.setUniformComponent(2, wB_bias.z);
  if (comp == 6) {
    h.setUniformComponent(3, wB_bias.x);
    h.setUniformComponent(4, wB_bias.y);
    h.setUniformComponent(5, wB_bias.z);
  }
  
  mB_bias.addToField(h);

  auto strayFields = magnet->getStrayFields();
  for (auto strayField : strayFields) {
    // Avoid the demag field, we only want external fields
    if (strayField->source() == magnet)
      continue;
    strayField->addToField(h);
  }
  return h;
}

Field evalZeemanEnergyDensity(const Ferromagnet* magnet) {
  if (externalFieldAssuredZero(magnet))
    return Field(magnet->system(), magnet->magnetization()->ncomp() / 3, 0.0);
  return evalEnergyDensity(magnet, evalExternalField(magnet), 1.0);
}

real zeemanEnergy(const Ferromagnet* magnet, const bool sub2) {
  if (externalFieldAssuredZero(magnet))
    return 0.0;

  real edens;
  if (!sub2) 
    edens = zeemanEnergyDensityQuantity(magnet).average()[0];
  else 
    edens = zeemanEnergyDensityQuantity(magnet).average()[1];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity externalFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalExternalField, magnet->magnetization()->ncomp(), "external_field", "T");
}

FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalZeemanEnergyDensity, magnet->magnetization()->ncomp() / 3,
                          "zeeman_energy_density", "J/m3");
}

FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = sub2 ? "zeeman_energy2" : "zeeman_energy";
  return FM_ScalarQuantity(magnet, zeemanEnergy, sub2, name, "J");
}