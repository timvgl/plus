#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

bool afmExchangeAssuredZero(const Antiferromagnet*);

Field evalAFMExchangeField(const Antiferromagnet*, const Ferromagnet*);
Field evalAFMExchangeEnergyDensity(const Antiferromagnet*, const Ferromagnet*);
real evalAFMExchangeEnergy(const Antiferromagnet*, const Ferromagnet*);

AFM_FieldQuantity AFM_exchangeFieldQuantity(const Antiferromagnet*, const Ferromagnet*);
AFM_FieldQuantity AFM_exchangeEnergyDensityQuantity(const Antiferromagnet*, const Ferromagnet*);
AFM_ScalarQuantity AFM_exchangeEnergyQuantity(const Antiferromagnet*, const Ferromagnet*);

// returns the maximal angle between exchange coupled cells
//real evalMaxAngle(const Ferromagnet*, const bool sub2);
//FM_ScalarQuantity maxAngle(const Ferromagnet*, const bool sub2);




