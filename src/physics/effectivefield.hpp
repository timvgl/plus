#pragma once

#include "ferromagnetquantity.hpp"
#include "handler.hpp"

class Ferromagnet;
class Field;

Field evalEffectiveField(const Ferromagnet*);

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet*);
