#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Field;

Field evalTotalMag(const Antiferromagnet*);
AFM_FieldQuantity totalMagnetizationQuantity(const Antiferromagnet*);