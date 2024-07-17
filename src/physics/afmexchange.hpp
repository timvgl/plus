#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

bool afmExchangeAssuredZero(const Ferromagnet*);

Field evalAFMExchangeField(const Ferromagnet*);
Field evalAFMExchangeEnergyDensity(const Ferromagnet*);
real evalAFMExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity AFMexchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity AFMexchangeEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity AFMexchangeEnergyQuantity(const Ferromagnet*);

// returns the (maximal) angle between magnetization vectors in the same cell
// which are coupled by the intracell exchange interaction.
Field evalAngleField(const Antiferromagnet*);
real evalMaxAngle(const Antiferromagnet*);

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet*);
AFM_ScalarQuantity maxAngle(const Antiferromagnet*);




