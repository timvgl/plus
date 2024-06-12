#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

bool afmExchangeAssuredZero(const Antiferromagnet*);

Field evalAFMExchangeField(const Antiferromagnet*, const Ferromagnet*);
//Field evalExchangeEnergyDensity(const Ferromagnet*);
//real evalExchangeEnergy(const Ferromagnet*, const bool sub2);

AFM_FieldQuantity AFM_exchangeFieldQuantity(const Antiferromagnet*, const Ferromagnet*);
//FM_FieldQuantity exchangeEnergyDensityQuantity(const Ferromagnet*);
//FM_ScalarQuantity exchangeEnergyQuantity(const Ferromagnet*, const bool sub2);

// returns the maximal angle between exchange coupled cells
//real evalMaxAngle(const Ferromagnet*, const bool sub2);
//FM_ScalarQuantity maxAngle(const Ferromagnet*, const bool sub2);
