#include "bufferpool.hpp"
#include "cudaerror.hpp"

BufferPool bufferPool;

BufferPool::~BufferPool() {
  // TODO: clean up buffers
}

void* BufferPool::allocate(int size) {
  void* ptr;
  if (pool_[size].size() == 0) {
    checkCudaError(cudaMalloc((void**)&ptr, size));
  } else {
    ptr = pool_[size].back();
    pool_[size].pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
}

void BufferPool::free(void** ptr) {
  inUse_.erase(*ptr);
  checkCudaError(cudaFree(*ptr));
  *ptr = nullptr;
}

void BufferPool::recycle(void** ptr) {
  auto inUseIt = inUse_.find(*ptr);
  int size = inUseIt->second;
  inUse_.erase(inUseIt);
  pool_[size].push_back(*ptr);
  *ptr = nullptr;
}