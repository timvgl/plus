#pragma once

#include "datatypes.hpp"
#include "grid.hpp"
#include "quantity.hpp"

class Field;
class CuParameter;

class Parameter : public Quantity {
 public:
  Parameter(Grid grid, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(Field* values);

  bool isUniform() const;
  bool isZero() const;
  int ncomp() const;
  Grid grid() const;
  void evalIn(Field *) const;

  CuParameter cu() const;

 private:
  const Grid grid_;
  real uniformValue_;
  Field* field_;
};

struct CuParameter {
  const Grid grid;
  const real uniformValue;
  real* valuesPtr;

  __device__ bool isUniform() const;
  __device__ real valueAt(int idx) const;
  __device__ real valueAt(int3 coo) const;
};

__device__ inline bool CuParameter::isUniform() const {
  return !valuesPtr;
}

__device__ inline real CuParameter::valueAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return valuesPtr[idx];
}

__device__ inline real CuParameter::valueAt(int3 coo) const {
  return valueAt(grid.coord2index(coo));
}