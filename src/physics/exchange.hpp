#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool exchangeAssuredZero(const Ferromagnet*);
Field evalExchangeField(const Ferromagnet*);
Field evalExchangeEnergyDensity(const Ferromagnet*);
real evalExchangeEnergy(const Ferromagnet*, const bool sub2);

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet*, const bool sub2);

// returns the maximal angle between exchange coupled cells
real evalMaxAngle(const Ferromagnet*, const bool sub2);
FM_ScalarQuantity maxAngle(const Ferromagnet*, const bool sub2);
