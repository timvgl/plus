#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

Field evalNeelvector(const Ferromagnet*);
FM_FieldQuantity neelVectorQuantity(const Ferromagnet*);