#pragma once
#include "datatypes.hpp"
#include "gpubuffer.hpp"

using GVec = GpuBuffer<real>;

GVec add(const GVec&, const GVec&);
GVec add(real a1, const GVec& v1, real a2, const GVec& v2);
real dotSum(const GVec&, const GVec&);
real maxAbsValue(const GVec&);
