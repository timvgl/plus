#pragma once

#include <vector>
#include <memory>

#include "datatypes.hpp"
#include "grid.hpp"

class CuField;

class Field {
 public:
  Field(Grid grid, int nComponents);
  ~Field();

  Grid grid() const;
  int ncomp() const;
  real* devptr(int comp) const;

  void getData(real* buffer) const;
  void setData(real* buffer);
  void setUniformComponent(real value, int comp);
  void makeZero();

  void copyFrom(const Field*);
  std::unique_ptr<Field> newCopy() const;

  CuField cu() const;

  const int ncomp_;
  const Grid grid_;

 private:
  std::vector<real*> devptrs_;
  real** devptr_devptrs_;
};

struct CuField {
  const Grid grid;
  const int ncomp;
  real** ptrs;

  __device__ bool cellInGrid(int) const;
  __device__ bool cellInGrid(int3) const;

  __device__ real valueAt(int idx, int comp = 0) const;
  __device__ real valueAt(int3 coo, int comp = 0) const;

  __device__ real3 vectorAt(int idx) const;
  __device__ real3 vectorAt(int3 coo) const;

  __device__ void setValueInCell(int idx, int comp, real value);
  __device__ void setVectorInCell(int idx, real3 vec);
};

__device__ inline bool CuField::cellInGrid(int idx) const {
  return grid.cellInGrid(idx);
}

__device__ inline bool CuField::cellInGrid(int3 coo) const {
  return grid.cellInGrid(coo);
}

__device__ inline real CuField::valueAt(int idx, int comp) const {
  return ptrs[comp][idx];
}

__device__ inline real CuField::valueAt(int3 coo, int comp) const {
  return valueAt(grid.coord2index(coo), comp);
}

__device__ inline real3 CuField::vectorAt(int idx) const {
  return real3{ptrs[0][idx], ptrs[1][idx], ptrs[2][idx]};
}

__device__ inline real3 CuField::vectorAt(int3 coo) const {
  return vectorAt(grid.coord2index(coo));
}

__device__ inline void CuField::setValueInCell(int idx, int comp, real value) {
  ptrs[comp][idx] = value;
}

__device__ inline void CuField::setVectorInCell(int idx, real3 vec) {
  ptrs[0][idx] = vec.x;
  ptrs[1][idx] = vec.y;
  ptrs[2][idx] = vec.z;
}