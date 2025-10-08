#include <iomanip>
#include <iostream>
#include <map>
#include <utility>

#include "cudaerror.hpp"
#include "gpumemorypool.hpp"

GpuMemoryPool memoryPool;

/* GpuMemoryPool::~GpuMemoryPool() {
  for (const auto& poolEntry : pool_)
    for (auto& ptr : poolEntry.second) {
      cudaError_t freeErr = cudaFree(ptr);
       if (freeErr == cudaErrorCudartUnloading)
          continue; // Expect CUDA driver to be shutting down
      checkCudaError(freeErr);
    }
} */

GpuMemoryPool::~GpuMemoryPool() {
  std::lock_guard<std::mutex> lk(m_);
  for (auto& kv : pool_) {
    auto& vec = kv.second;
    for (void* ptr : vec) {
      cudaError_t st = cudaFree(ptr);
      if (st == cudaErrorCudartUnloading) continue;
      checkCudaError(st);
    }
  }
  pool_.clear();
  inUse_.clear();
}

/* void* GpuMemoryPool::allocate(size_t size) {
  void* ptr;
  if (pool_[size].size() == 0) {
    checkCudaError(cudaMalloc(reinterpret_cast<void**>(&ptr), size));
  } else {
    ptr = pool_[size].back();
    pool_[size].pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
} */

void* GpuMemoryPool::allocate(size_t size) {
  std::lock_guard<std::mutex> lk(m_);
  void* ptr = nullptr;
  auto& freeList = pool_[size];
  if (freeList.empty()) {
    checkCudaError(cudaMalloc(&ptr, size));
  } else {
    ptr = freeList.back();
    freeList.pop_back();
  }
  inUse_[ptr] = size;
  return ptr;
}

/* void GpuMemoryPool::free(void** ptr) {
  inUse_.erase(*ptr);
  checkCudaError(cudaFree(*ptr));
  *ptr = nullptr;
} */

void GpuMemoryPool::free(void** pptr) {
  if (!pptr || !*pptr) return;
  void* ptr = *pptr;

  {
    std::lock_guard<std::mutex> lk(m_);
    auto it = inUse_.find(ptr);
    if (it != inUse_.end()) inUse_.erase(it);
  }

  cudaError_t st = cudaFree(ptr);
  if (st != cudaErrorCudartUnloading) checkCudaError(st);
  *pptr = nullptr;
}

/* void GpuMemoryPool::recycle(void** ptr) {
  auto inUseIt = inUse_.find(*ptr);
  int size = inUseIt->second;
  inUse_.erase(inUseIt);
  pool_[size].push_back(*ptr);
  *ptr = nullptr;
} */

void GpuMemoryPool::recycle(void** pptr) {
  if (!pptr || !*pptr) return;
  void* ptr = *pptr;

  size_t size = 0;
  {
    std::lock_guard<std::mutex> lk(m_);
    auto it = inUse_.find(ptr);
    if (it == inUse_.end()) {
      *pptr = nullptr;
      return;
    }
    size = it->second;
    inUse_.erase(it);
    pool_[size].push_back(ptr);
  }

  *pptr = nullptr;
}

/* void GpuMemoryPool::printInfo() const {
  // memoryUse map:
  //   key           memory block size
  //   value.first   in use count
  //   value.second  in pool count
  std::map<size_t, std::pair<int, int>> memoryUse;

  // count number of used memory blocks (for each memory block size seperately)
  for (auto u : inUse_)
    memoryUse[u.second].first++;

  // get number of memory blocks in the pool for each memory block size
  for (const auto& p : pool_)
    memoryUse[p.first].second = p.second.size();

  int totalMemUsed = 0;
  for (auto b : memoryUse) {
    totalMemUsed += b.first * (b.second.first + b.second.second);
  }

  std::cout << "GPU MEMORY POOL INFORMATION:" << std::endl;
  int colwidth = 10;
  std::cout << std::setw(colwidth) << "size(bytes)";
  std::cout << std::setw(colwidth) << "in use";
  std::cout << std::setw(colwidth) << "free" << std::endl;
  for (auto b : memoryUse) {
    std::cout << std::setw(colwidth) << b.first;
    std::cout << std::setw(colwidth) << b.second.first;
    std::cout << std::setw(colwidth) << b.second.second << std::endl;
  }
  std::cout << "Total used GPU memory:  " << totalMemUsed << " bytes"
            << std::endl;
} */

void GpuMemoryPool::printInfo() const {
  std::lock_guard<std::mutex> lk(m_);

  std::map<size_t, std::pair<int,int>> memoryUse;
  for (const auto& u : inUse_) memoryUse[u.second].first++;
  for (const auto& p : pool_) memoryUse[p.first].second = (int)p.second.size();

  int totalMemUsed = 0;
  for (auto& b : memoryUse) totalMemUsed += (int)b.first * (b.second.first + b.second.second);

  std::cout << "GPU MEMORY POOL INFORMATION:\n";
  int colwidth = 10;
  std::cout << std::setw(colwidth) << "size(bytes)"
            << std::setw(colwidth) << "in use"
            << std::setw(colwidth) << "free" << std::endl;
  for (auto& b : memoryUse) {
    std::cout << std::setw(colwidth) << b.first
              << std::setw(colwidth) << b.second.first
              << std::setw(colwidth) << b.second.second << std::endl;
  }
  std::cout << "Total used GPU memory:  " << totalMemUsed << " bytes\n";
}
