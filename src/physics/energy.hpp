#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

Field evalTotalEnergyDensity(const Ferromagnet*);
real evalTotalEnergy(const Ferromagnet*);

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet *);
FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet *);