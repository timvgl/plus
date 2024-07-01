#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool afmExchangeAssuredZero(const Ferromagnet*);

Field evalAFMExchangeField(const Ferromagnet*);
Field evalAFMExchangeEnergyDensity(const Ferromagnet*);
real evalAFMExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity AFMexchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity AFMexchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity AFMexchangeEnergyQuantity(const Ferromagnet*);

// returns the maximal angle between exchange coupled cells
//real evalMaxAngle(const Ferromagnet*, const bool sub2);
//FM_ScalarQuantity maxAngle(const Ferromagnet*, const bool sub2);




