#include <stdexcept>

#include "cudalaunch.hpp"
#include "vec.hpp"

__global__ void k_add(real* y, real a1, real* x1, real a2, real* x2, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;
  y[idx] = a1 * x1[idx] + a2 * x2[idx];
}

GVec add(real a1, const GVec& x1, real a2, const GVec& x2) {
  if (x1.size() != x2.size()) {
    throw std::invalid_argument(
        "Arrays can not be added together because their dimensions do not "
        "match");
  }
  int N = x1.size();
  GVec y(N);
  cudaLaunch(N, k_add, y.get(), a1, x1.get(), a2, x2.get(), N);
  return y;
}

GVec add(const GVec& x1, const GVec& x2) {
  return add(1, x1, 1, x2);
}

__global__ void k_maxAbsValue(real* result, real* x, int N) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < N; i += BLOCKDIM) {
    real value = abs(x[i]);
    threadValue = value > threadValue ? value : threadValue;
  }
  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      if (sdata[tid + s] > sdata[tid])
        sdata[tid] = sdata[tid + s];
    __syncthreads();
  }

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real maxAbsValue(const GVec& x) {
  if (x.size() == 0)
    return 0.0;

  real* d_result = (real*)bufferPool.allocate(sizeof(real));
  cudaLaunchReductionKernel(k_maxAbsValue, d_result, x.get(), (int)x.size());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result, 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  bufferPool.recycle((void**)&d_result);
  return result;
}

__global__ void k_dotSum(real* result, real* f, real* g, int N) {
  __shared__ real sdata[BLOCKDIM];
  int tid = threadIdx.x;

  real threadValue = 0.0;
  for (int i = tid; i < N; i += BLOCKDIM)
    threadValue += f[i] * g[i];

  sdata[tid] = threadValue;
  __syncthreads();

  // Reduce the block
  for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real dotSum(const GVec& x1, const GVec& x2) {
  if (x1.size() != x2.size()) {
    throw std::invalid_argument(
        "Arrays can not be added together because their dimensions do not "
        "match");
  }

  if (x1.size() == 0)
    return 0.0;

  real* d_result = (real*)bufferPool.allocate(sizeof(real));
  cudaLaunchReductionKernel(k_dotSum, d_result, x1.get(), x2.get(),
                            (int)x1.size());
  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result, sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  bufferPool.recycle((void**)&d_result);
  return result;
}