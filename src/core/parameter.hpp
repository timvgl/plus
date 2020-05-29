#pragma once

#include "datatypes.hpp"
#include "fieldquantity.hpp"
#include "grid.hpp"

class Field;
class CuParameter;

class Parameter : public FieldQuantity {
 public:
  Parameter(Grid grid, real value = 0.0);
  ~Parameter();

  void set(real value);
  void set(Field* values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  Grid grid() const;
  Field eval() const;

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

class CuVectorParameter;

class VectorParameter : public FieldQuantity {
 public:
  VectorParameter(Grid grid, real3 value = {0.0, 0.0, 0.0});
  ~VectorParameter();

  void set(real3 value);
  void set(Field* values);

  bool isUniform() const;
  bool assuredZero() const;
  int ncomp() const;
  Grid grid() const;
  Field eval() const;

  CuVectorParameter cu() const;

 private:
  const Grid grid_;
  real3 uniformValue_;
  Field* field_;
};

struct CuVectorParameter {
  const Grid grid;
  const real3 uniformValue;
  real* xValuesPtr;
  real* yValuesPtr;
  real* zValuesPtr;

  __device__ bool isUniform() const;
  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;
};

__device__ inline bool CuVectorParameter::isUniform() const {
  return !xValuesPtr;
}

__device__ inline real3 CuVectorParameter::vectorAt(int idx) const {
  if (isUniform())
    return uniformValue;
  return {xValuesPtr[idx], yValuesPtr[idx], zValuesPtr[idx]};
}

__device__ inline real3 CuVectorParameter::vectorAt(int3 coo) const {
  return vectorAt(grid.coord2index(coo));
}
