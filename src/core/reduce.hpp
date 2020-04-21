#pragma once

#include"datatypes.hpp"

#include<vector>

class Field;

real maxVecNorm(Field*);
real dotSum(Field*, Field*);
std::vector<real> fieldAverage(Field*);