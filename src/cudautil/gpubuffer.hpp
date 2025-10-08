#pragma once

#include <vector>

#include "cudaerror.hpp"
#include "cudastream.hpp"
#include "gpumemorypool.hpp"

enum class ZeroInitMode { Default, FFT };

template <typename T>
class GpuBuffer {
 public:
  // _______________________________________________________________CONSTRUCTORS

  GpuBuffer() {}                     /** Empty gpu buffer. */
  explicit GpuBuffer(size_t N); 
  explicit GpuBuffer(size_t N, cudaStream_t stream_);      /** Gpu buffer for N objects */
  GpuBuffer(const GpuBuffer& other); /** Copy constructor */
  GpuBuffer(GpuBuffer&& other);      /** Move constructor */

  /**
   * Construct a gpu buffer from a std::vector.
   * The size of the gpu buffer matches the size of the vector.
   * The values are copied from the vector on the host to the gpu.
   */
  explicit GpuBuffer(const std::vector<T>& other);
  explicit GpuBuffer(const std::vector<T>& other, cudaStream_t stream_);

  /** Gpubuffer with size N, and copy N values from data pointer. */
  GpuBuffer(int N, T* data);
  GpuBuffer(int N, T* data, cudaStream_t stream_);

  // ________________________________________________________________DESTRUCTION

  /** Destroy and release allocated memory back to the memory pool. */
  ~GpuBuffer() { recycle(); }

  // ____________________________________________________________________GETTERS

  size_t size() const { return size_; } /** Number of elements in buffer. */
  T* get() const { return ptr_; }       /** Device ptr of the buffer. */
  std::vector<T> getData() const;                           /** Get copy of data on the host. */

  /**
   * Copy the data from the gpubuffer to a newly allocated array on the
   * host. WARNING: This function returns a raw pointer, the user is
   * responsible to free the memory when done!
   */
  T* getHostCopy() const;

  cudaStream_t getStream() const { return stream_; }

  // ___________________________________________________________________MODIFIERS

  GpuBuffer<T>& operator=(const GpuBuffer<T>& other); /** Copy assignment */
  GpuBuffer<T>& operator=(GpuBuffer<T>&& other);      /** Move assignment */

  void recycle(); /** Empty the buffer and releases memory to the pool. */

  /**
   * Resizes the buffer. The buffer content will only be preserved if the new
   * size matches the old size.
   */
  void allocate(size_t size);

  // _________________________________________________________PRIVATE_DATAMEMBERS
 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
  cudaStream_t stream_ = nullptr; // non-owning, borrowed
};

//-------------------------------------------------------------------------------
// IMPLEMENTATION OF GPUBUFFER METHODS
//-------------------------------------------------------------------------------
template <class T>
inline GpuBuffer<T>::GpuBuffer(size_t N) :
    stream_(getCudaStream())
{
  allocate(N);
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(size_t N, cudaStream_t stream_) :
    stream_(stream_)
{
  allocate(N);
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(const std::vector<T>& other) : 
    stream_(getCudaStream())
{
  allocate(other.size());
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, other.data(), size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(const std::vector<T>& other, cudaStream_t stream_) : 
    stream_(stream_)
{
  allocate(other.size());
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, other.data(), size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(int N, T* data) :
    stream_(getCudaStream())
{
  allocate(N);
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, data, size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(int N, T* data, cudaStream_t stream_) :
    stream_(stream_)
{
  allocate(N);
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, data, size_ * sizeof(T),
                                   cudaMemcpyHostToDevice, stream_));
  }
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(const GpuBuffer& other) {
  stream_ = other.stream_;
  allocate(other.size());
  if (size_ == 0)
    return;
  checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                 cudaMemcpyDeviceToDevice, other.stream_));
}

template <class T>
inline GpuBuffer<T>::GpuBuffer(GpuBuffer&& other) {
  recycle();
  ptr_ = other.ptr_;
  size_ = other.size_;
  stream_ = other.stream_;
  other.ptr_ = nullptr;
  other.size_ = 0;
  other.stream_ = nullptr;
}

template <class T>
inline GpuBuffer<T>& GpuBuffer<T>::operator=(const GpuBuffer<T>& other) {
  stream_ = other.stream_;
  allocate(other.size());
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(ptr_, other.ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToDevice, other.stream_));
  }
  return *this;
}

template <class T>
inline GpuBuffer<T>& GpuBuffer<T>::operator=(GpuBuffer<T>&& other) {
  recycle();
  ptr_ = other.ptr_;
  size_ = other.size_;
  stream_ = other.stream_; 
  other.ptr_ = nullptr;
  other.size_ = 0;
  other.stream_ = nullptr;
  return *this;
}

template <class T>
inline void GpuBuffer<T>::allocate(size_t size) {
  if (size_ == size)
    return;

  recycle();

  if (size > 0) {
    ptr_ = reinterpret_cast<T*>(memoryPool.allocate(size * sizeof(T)));
  } else {
    ptr_ = nullptr;
  }

  size_ = size;
}

template <class T>
inline void GpuBuffer<T>::recycle() {
  if (ptr_)
    memoryPool.recycle(reinterpret_cast<void**>(&ptr_));
  ptr_ = nullptr;
  size_ = 0;
}


template <class T>
inline T* GpuBuffer<T>::getHostCopy() const {
  T* data = new T[size_];
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(data, ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_));
  }
  return data;
}

template <class T>
inline std::vector<T> GpuBuffer<T>::getData() const {
  std::vector<T> vec(size_);
  if (size_ > 0) {
    checkCudaError(cudaMemcpyAsync(vec.data(), ptr_, size_ * sizeof(T),
                                   cudaMemcpyDeviceToHost, stream_));
  }
  return vec;
}

// Explicit bool specialization for GpuBuffer<bool>::getData().
// This specialization is needed because unlike othter element types,
// std::vector<bool> does not contain a contiguous array of elements. Hence, the
// boolean array data on the gpu can not directly copied into an
// std::vector<bool>. We need a temporary bool buffer to copy into.
template <>
inline std::vector<bool> GpuBuffer<bool>::getData() const {
  bool* buf = getHostCopy();
  std::vector<bool> data(size_);
  for (int i = 0; i < (int)size_; i++) {
    data[i] = buf[i];
  }
  delete[] buf;
  return data;
}
