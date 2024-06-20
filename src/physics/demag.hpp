#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

bool demagFieldAssuredZero(const Ferromagnet*);

Field evalDemagField(const Ferromagnet*);
Field evalAFMDemagField(const Antiferromagnet*, const Ferromagnet*);

Field evalDemagEnergyDensity(const Ferromagnet*);
real evalDemagEnergy(const Ferromagnet*);

FM_FieldQuantity demagFieldQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_demagFieldQuantity(const Antiferromagnet*, const Ferromagnet*);
FM_FieldQuantity demagEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity demagEnergyQuantity(const Ferromagnet*);
