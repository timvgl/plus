#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

bool conductivityTensorAssuredZero(const Ferromagnet*);
Field evalConductivityTensor(const Ferromagnet*);
FM_FieldQuantity conductivityTensorQuantity(const Ferromagnet*);