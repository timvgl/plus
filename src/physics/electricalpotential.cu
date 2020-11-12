#include "electricalpotential.hpp"
#include "ferromagnet.hpp"
#include "poissonsolver.hpp"

bool electricalPotentialAssuredZero(const Ferromagnet* magnet) {
  return false;  // TODO: return true if no potential difference applied
}

Field evalElectricalPotential(const Ferromagnet* magnet) {
  Field pot(magnet->grid(), 1);
  if (electricalPotentialAssuredZero(magnet)) {
    pot.makeZero();
    return pot;
  }

  pot = magnet->poissonsolver.solve();

  return pot;
}

FM_FieldQuantity electricalPotentialQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElectricalPotential, 1,
                          "electrical_potential", "V");
}