#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool unianisotropyAssuredZero(const Ferromagnet*);
bool cubicanisotropyAssuredZero(const Ferromagnet*);
Field evalAnisotropyField(const Ferromagnet*);
Field evalAnisotropyEnergyDensity(const Ferromagnet*);
real evalAnisotropyEnergy(const Ferromagnet*, const bool sub2);

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet*);
FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet*, const bool sub2);
FM_ScalarQuantity anisotropyEnergyQuantity_sub2(const Ferromagnet*);