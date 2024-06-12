#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalEffectiveField(const Ferromagnet*);
Field evalAFMEffectiveField(const Antiferromagnet*, const Ferromagnet*);

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_effectiveFieldQuantity(const Antiferromagnet*, const Ferromagnet*);
