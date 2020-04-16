#include <iostream>
#include <stdexcept>

#include "bufferpool.hpp"
#include "cudaerror.hpp"
#include "cudastream.hpp"
#include "field.hpp"

Field::Field(Grid grid, int nComponents)
    : grid_(grid), ncomp_(nComponents), devptrs_(nComponents) {
  if (ncomp_ <= 0) {
    throw std::invalid_argument(
        "Number of components should be larger than zero");
  }

  for (auto& p : devptrs_) {
    p = bufferPool.allocate(grid.ncells());
  }
  checkCudaError(cudaMalloc((void**)&devptr_devptrs_, ncomp_ * sizeof(real*)));
  checkCudaError(cudaMemcpyAsync(devptr_devptrs_, &devptrs_[0],
                            ncomp_ * sizeof(real*), cudaMemcpyHostToDevice,
                            getCudaStream()));
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

real* Field::devptr(int comp) const {
  return devptrs_.at(comp);
}

void Field::getData(real* buffer) const {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpyAsync(bufferComponent, devptrs_[c],
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyDeviceToHost, getCudaStream()));
  }
}

void Field::setData(real* buffer) {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpyAsync(devptrs_[c], bufferComponent,
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyHostToDevice, getCudaStream()));
  }
}

void Field::copyFrom(const Field* src) {
  // TODO: throw error if field dimensions mismatch
  for (int c = 0; c < ncomp_; c++) {
    checkCudaError(cudaMemcpyAsync(devptrs_[c], src->devptrs_[c],
                              grid_.ncells() * sizeof(real),
                              cudaMemcpyDeviceToDevice, getCudaStream()));
  }
}

CuField Field::cu() const {
  return CuField{grid_, ncomp_, devptr_devptrs_};
}