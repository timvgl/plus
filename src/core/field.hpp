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
  int datasize() const;

  void getData(real* buffer) const;
  void setData(real* buffer);

  void copyFrom(const Field*);

 private:
  const Grid grid_;
  const int nComponents_;
  real* dataptr_;

 public:
  CuField* cu() const;

 private:
  CuField* cuField_;
};

class CuField {
 public:
  __host__ static CuField* create(Grid, int nComponents, real* dataptr);
  __device__ CuField(Grid, int nComponents, real* dataptr);

  __device__ Grid grid() const;
  __device__ int nComponents() const;

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

  real* dataptr;

 private:
  const Grid grid_;
  const int nComponents_;
};