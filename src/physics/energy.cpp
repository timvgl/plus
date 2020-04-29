#include "energy.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

TotalEnergyDensity::TotalEnergyDensity(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 1, "total_energy_density", "J/m3") {}

void TotalEnergyDensity::evalIn(Field* result) const {
    result->makeZero();
    ferromagnet_->anisotropyEnergyDensity()->addTo(result);
    ferromagnet_->exchangeEnergyDensity()->addTo(result);
    ferromagnet_->demagEnergyDensity()->addTo(result);
    ferromagnet_->zeemanEnergyDensity()->addTo(result);
}

TotalEnergy::TotalEnergy(Ferromagnet* ferromagnet)
    : FerromagnetScalarQuantity(ferromagnet, "total_energy", "J") {}

real TotalEnergy::eval() const {
  int ncells = ferromagnet_->grid().ncells();
  real edensAverage = ferromagnet_->totalEnergyDensity()->average()[0];
  real cellVolume = ferromagnet_->world()->cellVolume();
  return ncells * edensAverage * cellVolume;
}