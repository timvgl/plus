#pragma once

#include <vector>

#include "datatypes.hpp"

class Field;

real maxVecNorm(Field*);
real dotSum(Field*, Field*);
real fieldComponentAverage(Field*, int comp);
std::vector<real> fieldAverage(Field*);