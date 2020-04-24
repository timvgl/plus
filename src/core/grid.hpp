#pragma once

#include "datatypes.hpp"

class Grid {
 public:
  __host__ Grid(int3 size, int3 origin = {0, 0, 0});

  __host__ void setSize(int3);
  __host__ void setOrigin(int3);

  __device__ __host__ int3 size() const;
  __device__ __host__ int3 origin() const;
  __device__ __host__ int ncells() const;

  __device__ __host__ int3 index2coord(int idx) const;
  __device__ __host__ int coord2index(int3 coo) const;

  __device__ __host__ bool cellInGrid(int idx) const;
  __device__ __host__ bool cellInGrid(int3 coo) const;

  __device__ __host__ bool overlaps(Grid) const;

  __host__ friend bool operator==(const Grid& lhs, const Grid& rhs);
  __host__ friend bool operator!=(const Grid& lhs, const Grid& rhs);

 private:
  int3 size_;
  int3 origin_;
};

// -----------------------------------------------------------------------
// Implementation of inline member functions

__device__ __host__ inline int3 Grid::size() const {
  return size_;
}

__device__ __host__ inline int3 Grid::origin() const {
  return origin_;
}

__device__ __host__ inline int Grid::ncells() const {
  return size_.x * size_.y * size_.z;
}

__device__ __host__ inline int3 Grid::index2coord(int idx) const {
  return {
    x : origin_.x + idx % size_.x,
    y : origin_.y + (idx / size_.x) % size_.y,
    z : origin_.z + idx / (size_.x * size_.y)
  };
}

__device__ __host__ inline int Grid::coord2index(int3 coo) const {
  coo -= origin_;
  return coo.x + coo.y * size_.x + coo.z * size_.x * size_.y;
}

__device__ __host__ inline bool Grid::cellInGrid(int idx) const {
  return idx >= 0 && idx < ncells();
}

__device__ __host__ inline bool Grid::cellInGrid(int3 coo) const {
  coo -= origin_;
  return coo.x >= 0 && coo.x < size_.x && coo.y >= 0 && coo.y < size_.y &&
         coo.z >= 0 && coo.z < size_.z;
}

__device__ __host__ inline bool Grid::overlaps(Grid other) const {
  auto max = [](int a, int b) { return a > b ? a : b; };
  auto min = [](int a, int b) { return a < b ? a : b; };
  // TODO: min and max functions don't belong here

  int x1 = max(origin_.x, other.origin_.x);
  int y1 = max(origin_.y, other.origin_.y);
  int z1 = max(origin_.z, other.origin_.z);
  int x2 = min(origin_.x + size_.x, other.origin_.x + other.size_.x);
  int y2 = min(origin_.y + size_.y, other.origin_.y + other.size_.y);
  int z2 = min(origin_.z + size_.z, other.origin_.z + other.size_.z);
  return (x2 - x1) > 0 && (y2 - y1) > 0 && (z2 - z1) > 0;
}