#include <stdexcept>

#include "bufferpool.hpp"
#include "cudaerror.hpp"
#include "cudastream.hpp"
#include "datatypes.hpp"
#include "field.hpp"
#include "reduce.hpp"

#define BLOCKDIM 512

template <typename... Arguments>
void cudaLaunchReductionKernel(void (*kernelfunction)(Arguments...),
                               Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims(1);
  kernelfunction<<<gridDims, blockDims, 0, getCudaStream()>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}

__global__ void k_maxVecNorm(real* result, CuField f) {
  // Reduce to a block
  __shared__ real sdata[BLOCKDIM];
  int ncells = f.grid.ncells();
  int tid = threadIdx.x;
  real threadValue = 0.0;
  for (int i = tid; i < ncells; i += BLOCKDIM) {
    real3 cellVec = f.vectorAt(i);
    real cellNorm = norm(cellVec);
    if (cellNorm > threadValue) {
      threadValue = cellNorm;
    }
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
  // TODO: check if loop unrolling makes sense here

  // Set the result
  if (tid == 0)
    *result = sdata[0];
}

real maxVecNorm(Field* f) {
  if (f->ncomp() != 3) {
    throw std::runtime_error(
        "the input field of maxVecNorm should have 3 components");
  }

  real* d_result = bufferPool.allocate(1);
  cudaLaunchReductionKernel(k_maxVecNorm, d_result, f->cu());

  // copy the result to the host and return
  real result;
  checkCudaError(cudaMemcpyAsync(&result, d_result, 1 * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  bufferPool.recycle(d_result);
  return result;
}

__global__ void k_average(real* result, CuField f) {
  __shared__ real sdata[BLOCKDIM];
  int tid = threadIdx.x;
  int ncells = f.grid.ncells();
  int ncomp = f.ncomp;

  for (int c = 0; c < ncomp; c++) {
    // Reduce to a block
    real threadValue = 0.0;
    for (int i = tid; i < ncells; i += BLOCKDIM)
      threadValue += f.ptrs[c][i];
    sdata[tid] = threadValue;
    __syncthreads();

    // Reduce the block
    for (unsigned int s = BLOCKDIM / 2; s > 0; s >>= 1) {
      if (tid < s)
        sdata[tid] += sdata[tid + s];
      __syncthreads();
    }
    // TODO: check if loop unrolling makes sense here

    // Set the result
    if (tid == 0)
      result[c] = sdata[0] / ncells;
  }
}

std::vector<real> fieldAverage(Field* f) {
  real* d_result = bufferPool.allocate(f->ncomp());
  cudaLaunchReductionKernel(k_average, d_result, f->cu());
  // copy the result to the host and return
  std::vector<real> result(f->ncomp());
  checkCudaError(cudaMemcpyAsync(&result[0], d_result,
                                 f->ncomp() * sizeof(real),
                                 cudaMemcpyDeviceToHost, getCudaStream()));
  bufferPool.recycle(d_result);
  return result;
}