#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

// Returns the energy density of an effective field term.
//   edens = - prefactor * Msat * dot(m,h)
// The prefactor depends on the origin of the effective field term
Field evalEnergyDensity(const Ferromagnet*, const Field&, real prefactor);

Field evalTotalEnergyDensity(const Ferromagnet*);
real evalTotalEnergy(const Ferromagnet*);

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet*);
