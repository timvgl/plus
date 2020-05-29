#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

bool exchangeAssuredZero(const Ferromagnet *);
Field evalExchangeField(const Ferromagnet*);
Field evalExchangeEnergyDensity(const Ferromagnet*);
real evalExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity exchangeFieldQuantity(const Ferromagnet *);
FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet *);