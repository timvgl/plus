#include "energy.hpp"

#include "anisotropy.hpp"
#include "demag.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "interfacialdmi.hpp"
#include "world.hpp"
#include "zeeman.hpp"

Field evalTotalEnergyDensity(const Ferromagnet* magnet) {
  Field edens = evalDemagEnergyDensity(magnet);
  edens += evalAnisotropyEnergyDensity(magnet);
  edens += evalExchangeEnergyDensity(magnet);
  edens += evalZeemanEnergyDensity(magnet);
  edens += evalInterfacialDmiEnergyDensity(magnet);
  return edens;
}

real evalTotalEnergy(const Ferromagnet* magnet) {
  int ncells = magnet->grid().ncells();
  real edensAverage = totalEnergyDensityQuantity(magnet).average()[0];
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTotalEnergyDensity, 1,
                             "total_energy_density", "J/m3");
}

FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}