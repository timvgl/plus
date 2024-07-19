#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalFullMag(const Ferromagnet*);
FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);

Field evalAFMFullMag(const Antiferromagnet*);
AFM_FieldQuantity afmFullMagnetizationQuantity(const Antiferromagnet*);