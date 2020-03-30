#include <stdexcept>

#include "datatypes.hpp"
#include "grid.hpp"

Grid::Grid(int3 size, int3 origin) {
  setSize(size);
  setOrigin(origin);
}

void Grid::setSize(int3 size) {
  if (size.x <= 0 || size.y <= 0 || size.z <= 0) {
    throw std::invalid_argument("The grid size should be larger than 0");
  }
  size_ = size;
}

void Grid::setOrigin(int3 origin) {
  origin_ = origin;
}

int3 Grid::size() const {
  return size_;
}

int3 Grid::origin() const {
  return origin_;
}

int Grid::ncells() const {
  return size_.x * size_.y * size_.z;
}

int3 Grid::idx2coo(int idx) const {
  return {
    x : idx % size_.x,
    y : (idx / size_.x) % size_.y,
    z : idx / (size_.x * size_.y)
  };
}

int Grid::coo2idx(int3 coo) const {
  return coo.x + coo.y * size_.x + coo.z * size_.x * size_.y;
}

bool operator==(const Grid& lhs, const Grid& rhs) {
  return lhs.origin_ == rhs.origin_ && lhs.origin_ == rhs.origin_;
}

bool operator!=(const Grid& lhs, const Grid& rhs) {
  return lhs.origin_ != rhs.origin_ || lhs.origin_ != rhs.origin_;
}