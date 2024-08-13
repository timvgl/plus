#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Field;

Field evalNeelvector(const Antiferromagnet*);
AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet*);