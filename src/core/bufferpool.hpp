#pragma once
#include <map>
#include <vector>

class BufferPool {
 public:
  BufferPool() = default;
  BufferPool(BufferPool const&) = delete;
  void operator=(BufferPool const&) = delete;

  ~BufferPool();

  void* allocate(int size);
  void free(void**);
  void recycle(void**);

 private:
  std::map<int, std::vector<void*>> pool_;
  std::map<void*, int> inUse_;
};

extern BufferPool bufferPool;

template <typename T>
class GpuPtr {
 public:
  GpuPtr(int N) {allocate(N);}
  GpuPtr() : ptr(nullptr) {}
  ~GpuPtr() { recycle(); }

  // disable copy constructor
  GpuPtr(const GpuPtr&) = delete;

  // Move constructor
  GpuPtr(GpuPtr&& other) {
    ptr = other.ptr;
    other.ptr = nullptr;
  }

  // Move assignment
  GpuPtr<T>& operator=(GpuPtr<T>&& other){
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }

  // disable assignment operator
  GpuPtr<T>& operator=(const GpuPtr<T>&) = delete;

  void allocate(int N) {
    recycle();
    ptr = (T*)bufferPool.allocate(N * sizeof(T));
  }

  void recycle() {
    if (ptr)
      bufferPool.recycle((void**)&ptr);
    ptr = nullptr;
  }

  T* get() const { return ptr; }

 private:
  T* ptr;
};