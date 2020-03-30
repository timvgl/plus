#include <iostream>
#include <stdexcept>

#include "cudaerror.hpp"
#include "field.hpp"

Field::Field(Grid grid, int nComponents)
    : grid_(grid), nComponents_(nComponents) {
  if (nComponents_ <= 0) {
    throw std::invalid_argument(
        "Number of components should be larger than zero");
  }

  checkCudaError(cudaMalloc((void**)&dataptr_, datasize() * sizeof(real)));

  cuField_ = CuField::create(grid_, nComponents_, dataptr_);
}

Field::~Field() {
  cudaFree(dataptr_);
  cudaFree(cuField_);
}

Grid Field::grid() const {
  return grid_;
}

int Field::ncomp() const {
  return nComponents_;
}

int Field::datasize() const {
  int3 gs = grid_.size();
  return nComponents_ * gs.x * gs.y * gs.z;
}

void Field::getData(real* buffer) const {
  checkCudaError(cudaMemcpy(buffer, dataptr_, datasize() * sizeof(real),
                            cudaMemcpyDeviceToHost));
}

void Field::setData(real* buffer) {
  checkCudaError(cudaMemcpy(dataptr_, buffer, datasize() * sizeof(real),
                            cudaMemcpyHostToDevice));
}

void Field::copyFrom(const Field* src) {
  // TODO: throw error if field dimensions mismatch
  checkCudaError(cudaMemcpy(dataptr_, src->dataptr_, datasize() * sizeof(real),
                            cudaMemcpyDeviceToDevice));
}

CuField* Field::cu() const {
  return cuField_;
}

static __global__ void k_constructCuField(CuField* cuField,
                                          Grid grid,
                                          int nComponents,
                                          real* dataptr) {
  new (cuField) CuField(grid, nComponents, dataptr);
}

__host__ CuField* CuField::create(Grid grid, int nComponents, real* dataptr) {
  CuField* cuField;
  cudaMalloc((void**)&cuField, sizeof(CuField));
  k_constructCuField<<<1, 1>>>(cuField, grid, nComponents, dataptr);
  return cuField;
}

__device__ CuField::CuField(Grid grid, int nComponents, real* dataptr)
    : grid_(grid), nComponents_(nComponents), dataptr(dataptr) {}

__device__ Grid CuField::grid() const {
  return grid_;
}

__device__ int CuField::nComponents() const {
  return nComponents_;
}

__device__ bool CuField::cellInGrid(int idx) const {
  return idx >= 0 && idx < grid_.ncells();
}

__device__ bool CuField::cellInGrid(int3 coo) const {
  int3 gs = grid_.size();
  return coo.x >= 0 && coo.x < gs.x && coo.y >= 0 && coo.y < gs.y &&
         coo.z >= 0 && coo.z < gs.z;
}

__device__ bool CuField::cellInGrid() const {
  return cellInGrid(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ real CuField::cellValue(int idx, int comp) const {
  return dataptr[idx + grid_.ncells() * comp];
}

__device__ real CuField::cellValue(int3 coo, int comp) const {
  return cellValue(grid_.coo2idx(coo));
}

__device__ real CuField::cellValue(int comp) const {
  return cellValue(blockIdx.x * blockDim.x + threadIdx.x, comp);
}

__device__ real3 CuField::cellVector(int idx) const {
  return real3{dataptr[idx + grid_.ncells() * 0],
               dataptr[idx + grid_.ncells() * 1],
               dataptr[idx + grid_.ncells() * 2]};
}

__device__ real3 CuField::cellVector(int3 coo) const {
  return cellVector(grid_.coo2idx(coo));
}

__device__ real3 CuField::cellVector() const {
  return cellVector(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ void CuField::setCellValue(int comp, real value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dataptr[i + grid_.ncells() * comp] = value;
}

__device__ void CuField::setCellVector(real3 vec) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dataptr[i + grid_.ncells() * 0] = vec.x;
  dataptr[i + grid_.ncells() * 1] = vec.y;
  dataptr[i + grid_.ncells() * 2] = vec.z;
}