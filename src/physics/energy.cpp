#include "energy.hpp"

#include "anisotropy.hpp"
#include "demag.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"
#include "zeeman.hpp"

TotalEnergyDensity::TotalEnergyDensity(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 1, "total_energy_density", "J/m3") {
}

void TotalEnergyDensity::evalIn(Field* result) const {
  result->makeZero();
  AnisotropyEnergyDensity(ferromagnet_).addTo(result);
  ExchangeEnergyDensity(ferromagnet_).addTo(result);
  DemagEnergyDensity(ferromagnet_).addTo(result);
  ZeemanEnergyDensity(ferromagnet_).addTo(result);
}

TotalEnergy::TotalEnergy(Handle<Ferromagnet> ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "total_energy", "J") {}

real TotalEnergy::eval() const {
  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = TotalEnergyDensity(ferromagnet_).average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}