#include <iostream>
#include <stdexcept>

#include "cudaerror.hpp"
#include "field.hpp"

Field::Field(Grid grid, int nComponents)
    : grid_(grid), nComponents_(nComponents), devptrs_(nComponents) {
  if (nComponents_ <= 0) {
    throw std::invalid_argument(
        "Number of components should be larger than zero");
  }

  checkCudaError(cudaMalloc((void**)&dataptr_, datasize() * sizeof(real)));

  for (auto& p : devptrs_) {
    checkCudaError(cudaMalloc((void**)&p, grid_.ncells() * sizeof(real)));
  }

  checkCudaError(cudaMalloc((void**)&devptr_devptrs_, nComponents_ * sizeof(real*)));

  checkCudaError(cudaMemcpy(&devptrs_[0], devptr_devptrs_, nComponents_ * sizeof(real*),
                            cudaMemcpyDeviceToHost));
}

Field::~Field() {
  for (auto p : devptrs_) {
    cudaFree(p);
  }
  cudaFree(dataptr_);
  cudaFree(devptr_devptrs_);
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

CuField Field::cu() const{
  return CuField{dataptr_, grid_, nComponents_, devptr_devptrs_};
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
  return dataptr[idx + grid.ncells() * comp];
}

__device__ real CuField::cellValue(int3 coo, int comp) const {
  return cellValue(grid.coo2idx(coo), comp);
}

__device__ real CuField::cellValue(int comp) const {
  return cellValue(blockIdx.x * blockDim.x + threadIdx.x, comp);
}

__device__ real3 CuField::cellVector(int idx) const {
  return real3{dataptr[idx + grid.ncells() * 0],
               dataptr[idx + grid.ncells() * 1],
               dataptr[idx + grid.ncells() * 2]};
}

__device__ real3 CuField::cellVector(int3 coo) const {
  return cellVector(grid.coo2idx(coo));
}

__device__ real3 CuField::cellVector() const {
  return cellVector(blockIdx.x * blockDim.x + threadIdx.x);
}

__device__ void CuField::setCellValue(int comp, real value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dataptr[i + grid.ncells() * comp] = value;
}

__device__ void CuField::setCellVector(real3 vec) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dataptr[i + grid.ncells() * 0] = vec.x;
  dataptr[i + grid.ncells() * 1] = vec.y;
  dataptr[i + grid.ncells() * 2] = vec.z;
}