#pragma once

#include "antiferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalNeelvector(const Antiferromagnet*, const Ferromagnet*);
AFM_FieldQuantity neelVectorQuantity(const Antiferromagnet*);