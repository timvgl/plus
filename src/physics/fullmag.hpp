#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Field;

Field evalFullMag(const Antiferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);