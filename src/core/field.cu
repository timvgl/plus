#include <iostream>
#include <stdexcept>

#include "cudaerror.hpp"
#include "field.hpp"
#include "bufferpool.hpp"

Field::Field(Grid grid, int nComponents)
    : grid_(grid), ncomp_(nComponents), devptrs_(nComponents) {
  if (ncomp_ <= 0) {
    throw std::invalid_argument(
        "Number of components should be larger than zero");
  }

  for (auto& p : devptrs_) {
    p = bufferPool.allocate(grid.ncells());
  }
  checkCudaError(
      cudaMalloc((void**)&devptr_devptrs_, ncomp_ * sizeof(real*)));
  checkCudaError(cudaMemcpy(devptr_devptrs_,&devptrs_[0], 
                            ncomp_ * sizeof(real*),
                            cudaMemcpyHostToDevice));
}

Field::~Field() {
  for (auto p : devptrs_) {
    bufferPool.recycle(p);
  }
  cudaFree(devptr_devptrs_);
}

Grid Field::grid() const {
  return grid_;
}

int Field::ncomp() const {
  return ncomp_;
}

real * Field::devptr(int comp) const {
  return devptrs_.at(comp);
}

void Field::getData(real* buffer) const {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpy(bufferComponent, devptrs_[c],
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyDeviceToHost));
  }
}

void Field::setData(real* buffer) {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpy(devptrs_[c], bufferComponent,
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyHostToDevice));
  }
}

void Field::copyFrom(const Field* src) {
  // TODO: throw error if field dimensions mismatch
  for (int c = 0; c < ncomp_; c++) {
    checkCudaError(cudaMemcpy(devptrs_[c], src->devptrs_[c],
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyDeviceToDevice));
  }
}

CuField Field::cu() const {
  return CuField{grid_, ncomp_, devptr_devptrs_};
}

__device__ bool CuField::cellInGrid(int idx) const {
  return idx >= 0 && idx < grid.ncells();
}

__device__ bool CuField::cellInGrid(int3 coo) const {
  coo -= grid.origin();  // relative coordinate
  int3 gs = grid.size();
  return coo.x >= 0 && coo.x < gs.x && coo.y >= 0 && coo.y < gs.y &&
         coo.z >= 0 && coo.z < gs.z;
}

__device__ bool CuField::cellInGrid() const {
  return cellInGrid(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ real CuField::cellValue(int idx, int comp) const {
  return ptrs[comp][idx];
}

__device__ real CuField::cellValue(int3 coo, int comp) const {
  return cellValue(grid.coo2idx(coo), comp);
}

__device__ real CuField::cellValue(int comp) const {
  return cellValue(blockIdx.x * blockDim.x + threadIdx.x, comp);
}

__device__ real3 CuField::cellVector(int idx) const {
  return real3{ptrs[0][idx], ptrs[1][idx], ptrs[2][idx]};
}

__device__ real3 CuField::cellVector(int3 coo) const {
  return cellVector(grid.coo2idx(coo));
}

__device__ real3 CuField::cellVector() const {
  return cellVector(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ void CuField::setCellValue(int comp, real value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptrs[comp][i] = value;
}

__device__ void CuField::setCellVector(real3 vec) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptrs[0][i] = vec.x;
  ptrs[1][i] = vec.y;
  ptrs[2][i] = vec.z;
}