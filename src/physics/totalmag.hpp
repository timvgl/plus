#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalTotalMag(const Antiferromagnet*, const Ferromagnet*);
AFM_FieldQuantity totalMagnetizationQuantity(const Antiferromagnet*);