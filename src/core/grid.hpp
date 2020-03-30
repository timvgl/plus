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

  __device__ __host__ int3 idx2coo(int) const;
  __device__ __host__ int coo2idx(int3) const;

  __host__ friend bool operator==(const Grid& lhs, const Grid& rhs);
  __host__ friend bool operator!=(const Grid& lhs, const Grid& rhs);

 private:
  int3 size_;
  int3 origin_;
};