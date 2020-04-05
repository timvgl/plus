#pragma once

#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class CuField;

class Field {
 public:
  Field(Grid grid, int nComponents);
  ~Field();

  Grid grid() const;
  int ncomp() const;
  real * devptr(int comp) const;

  void getData(real* buffer) const;
  void setData(real* buffer);

  void copyFrom(const Field*);

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
  __device__ bool cellInGrid() const;

  __device__ real cellValue(int idx, int comp) const;
  __device__ real cellValue(int3 coo, int comp) const;
  __device__ real cellValue(int comp) const;

  __device__ real3 cellVector(int idx) const;
  __device__ real3 cellVector(int3 coo) const;
  __device__ real3 cellVector() const;

  __device__ void setCellValue(int comp, real value);
  __device__ void setCellVector(real3 vec);
};