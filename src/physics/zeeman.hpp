#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;
class Magnet;

bool externalFieldAssuredZero(const Magnet*);

Field calcExternalFields(const Magnet*, Field);
Field evalExternalField(const Ferromagnet*);
Field evalAFMExternalField(const Antiferromagnet*, const Ferromagnet*);
Field evalZeemanEnergyDensity(const Ferromagnet*);
real zeemanEnergy(const Ferromagnet*);

FM_FieldQuantity externalFieldQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_externalFieldQuantity(const Antiferromagnet*, const Ferromagnet*);
FM_FieldQuantity zeemanEnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity zeemanEnergyQuantity(const Ferromagnet*);
