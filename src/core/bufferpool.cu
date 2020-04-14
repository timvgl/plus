#include "bufferpool.hpp"
#include "cudaerror.hpp"
#include "datatypes.hpp"

BufferPool bufferPool;

real* BufferPool::allocate(int size) {
  real* ptr;
  if (pool_[size].size() == 0) {
    checkCudaError(cudaMalloc((void**)&ptr, size * sizeof(real)));
  } else {
    ptr = pool_[size].back();
    pool_[size].pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
}

void BufferPool::free(real*& ptr) {
  inUse_.erase(ptr);
  cudaFree(ptr);
  ptr = nullptr;
}

void BufferPool::recycle(real*& ptr) {
  auto inUseIt = inUse_.find(ptr);
  int size = inUseIt->second;
  inUse_.erase(inUseIt);
  pool_[size].push_back(ptr);
  ptr = nullptr;
}