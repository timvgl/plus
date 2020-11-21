#pragma once
#include "gpubuffer.hpp"

using lsReal = float;
using GVec = GpuBuffer<lsReal>;

GVec add(const GVec&, const GVec&);
GVec add(lsReal a1, const GVec& v1, lsReal a2, const GVec& v2);
lsReal dotSum(const GVec&, const GVec&);
lsReal maxAbsValue(const GVec&);