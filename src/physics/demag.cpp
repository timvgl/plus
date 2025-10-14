#include "demag.hpp"

#include "antiferromagnet.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

bool demagFieldAssuredZero(const Ferromagnet* magnet) {
  return !magnet->enableDemag || magnet->msat.assuredZero();
}

Field evalDemagField(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return Field(magnet->system(), 3, 0.0);
  if (magnet->isSublattice())
    return magnet->hostMagnet()->getStrayField(magnet->hostMagnet())->eval();
  return magnet->getStrayField(magnet)->eval();
}

Field evalDemagEnergyDensity(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  Field hdemag = evalDemagField(magnet);
  hdemag.ensureReadyOn(getCudaStream());
  Field edens = evalEnergyDensity(magnet, hdemag, 0.5);
  hdemag.markLastUse();
  edens.markLastUse();
  return edens;
}

real evalDemagEnergy(const Ferromagnet* magnet) {
  if (demagFieldAssuredZero(magnet))
    return 0.0;
  real edensAverage = demagEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edensAverage);
}

FM_FieldQuantity demagEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDemagEnergyDensity, 1,
                          "demag_energy_density", "J/m3");
}

FM_ScalarQuantity demagEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalDemagEnergy, "demag_energy", "J");
}
