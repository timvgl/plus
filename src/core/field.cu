#include <iostream>
#include <stdexcept>

#include "bufferpool.hpp"
#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"

Field::Field() : grid_({0, 0, 0}), ncomp_(0), devptr_devptrs_(nullptr) {}

Field::Field(Grid grid, int nComponents)
    : grid_(grid), ncomp_(nComponents), devptr_devptrs_(nullptr) {
  allocate();
}

Field::Field(const Field& other) : grid_(other.grid_), ncomp_(other.ncomp_) {
  allocate();
  copyFrom(&other);
}

// Move constructer
Field::Field(Field&& other) : grid_(other.grid_), ncomp_(other.ncomp_) {
  devptrs_ = other.devptrs_;
  devptr_devptrs_ = other.devptr_devptrs_;
  other.devptrs_.clear();
  other.devptr_devptrs_ = nullptr;
}

// Assignment operator
Field& Field::operator=(const Field& other) {
  if (this == &other) {
    return *this;
  }
  if (grid_ != other.grid_ || ncomp_ != other.ncomp_) {
    free();
    grid_ = other.grid_;
    ncomp_ = other.ncomp_;
    allocate();
  }
  copyFrom(&other);
  return *this;
}

// Evaluate quantity in this field
Field& Field::operator=(const FieldQuantity& q) {
  return operator=(q.eval());
}

// Move assignment operator
Field& Field::operator=(Field&& other) {
  grid_ = other.grid_;
  ncomp_ = other.ncomp_;
  devptrs_ = other.devptrs_;
  devptr_devptrs_ = other.devptr_devptrs_;
  other.devptrs_.clear();
  other.devptr_devptrs_ = nullptr;
  return *this;
}

Field::~Field() {
  free();
}

void Field::allocate() {
  if (ncomp_ == 0 || grid_.ncells() == 0)
    return;
  devptrs_.resize(ncomp_);
  for (auto& p : devptrs_) {
    p = bufferPool.allocate(grid_.ncells());
  }
  checkCudaError(cudaMalloc((void**)&devptr_devptrs_, ncomp_ * sizeof(real*)));
  checkCudaError(cudaMemcpyAsync(devptr_devptrs_, &devptrs_[0],
                                 ncomp_ * sizeof(real*), cudaMemcpyHostToDevice,
                                 getCudaStream()));
}

void Field::free() {
  for (auto p : devptrs_) {
    bufferPool.recycle(p);
  }
  if (devptr_devptrs_)
    cudaFree(devptr_devptrs_);
  devptr_devptrs_ = nullptr;
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

__global__ static void k_setComponent(CuField f, real value, int comp) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (!f.cellInGrid(idx))
    return;
  f.setValueInCell(idx, comp, value);
}

void Field::setUniformComponent(real value, int comp) {
  cudaLaunch(grid_.ncells(), k_setComponent, cu(), value, comp);
}

void Field::makeZero() {
  for (int comp = 0; comp < ncomp_; comp++)
    setUniformComponent(0.0, comp);
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

void Field::operator+=(const Field& x) {
  addTo(*this, 1, x);
}