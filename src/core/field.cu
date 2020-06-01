#include <iostream>
#include <stdexcept>

#include "bufferpool.hpp"
#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"

Field::Field() : grid_({0, 0, 0}), ncomp_(0) {}

Field::Field(Grid grid, int nComponents) : grid_(grid), ncomp_(nComponents) {
  allocate();
}

Field::Field(const Field& other) : grid_(other.grid_), ncomp_(other.ncomp_) {
  allocate();
  copyFrom(other);
}

Field::Field(Field&& other) : grid_(other.grid_), ncomp_(other.ncomp_) {
  devptrs_ = std::move(other.devptrs_);
  devptr_devptrs_ = std::move(other.devptr_devptrs_);
}

Field& Field::operator=(const Field& other) {
  if (this == &other)
    return *this;
  return *this = std::move(Field(other)); // moves a copy of other to this
}

Field& Field::operator=(const FieldQuantity& q) {
  return *this = std::move(q.eval());
}

Field& Field::operator=(Field&& other) {
  grid_ = other.grid_;
  ncomp_ = other.ncomp_;
  devptrs_ = std::move(other.devptrs_);
  devptr_devptrs_ = std::move(other.devptr_devptrs_);
  other.clear();
  return *this;
}

void Field::clear() {
  grid_ = Grid({0, 0, 0});
  ncomp_ = 0;
  free();
}

void Field::allocate() {
  free();

  if (empty())
    return;

  devptrs_.resize(ncomp_);
  for (auto& p : devptrs_)
    p.allocate(grid_.ncells());

  devptr_devptrs_.allocate(ncomp_);

  // The &devptrs_[0], is tricky here. This works because a DevPtr has only the
  // actual ptr as a datamember. TODO: check if this needs to be changed
  checkCudaError(cudaMemcpyAsync(devptr_devptrs_.get(), &devptrs_[0],
                                 ncomp_ * sizeof(real*), cudaMemcpyHostToDevice,
                                 getCudaStream()));
}

void Field::free() {
  devptrs_.clear();
  devptr_devptrs_.recycle();
}

CuField Field::cu() const {
  return CuField{grid_, ncomp_, devptr_devptrs_.get()};
}

void Field::getData(real* buffer) const {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpyAsync(bufferComponent, devptrs_[c].get(),
                                   grid_.ncells() * sizeof(real),
                                   cudaMemcpyDeviceToHost, getCudaStream()));
  }
}

void Field::setData(real* buffer) {
  for (int c = 0; c < ncomp_; c++) {
    real* bufferComponent = buffer + c * grid_.ncells();
    checkCudaError(cudaMemcpyAsync(devptrs_[c].get(), bufferComponent,
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

void Field::copyFrom(const Field& src) {
  // TODO: throw error if field dimensions mismatch
  for (int c = 0; c < ncomp_; c++) {
    checkCudaError(cudaMemcpyAsync(devptrs_[c].get(), src.devptrs_[c].get(),
                                   grid_.ncells() * sizeof(real),
                                   cudaMemcpyDeviceToDevice, getCudaStream()));
  }
}

Field& Field::operator+=(const Field& other) {
  addTo(*this, 1, other);
  return *this;
}

Field& Field::operator-=(const Field& other) {
  addTo(*this, -1, other);
  return *this;
}

Field& Field::operator+=(const FieldQuantity& q) {
  addTo(*this, 1, q.eval());
  return *this;
}

Field& Field::operator-=(const FieldQuantity& q) {
  addTo(*this, -1, q.eval());
  return *this;
}
