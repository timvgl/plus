#pragma once

#include "cudaerror.hpp"
#include "cudastream.hpp"
#include <string>

const int BLOCKDIM = 512;

// Source: https://devblogs.nvidia.com/cplusplus-11-in-cuda-variadic-templates/
template <typename... Arguments>
void cudaLaunch(std::string src, int N,
                void (*kernelfunction)(Arguments...),
                Arguments... args) {
  cudaFuncAttributes attr{};
  // NOTE: no overload ambiguity here; 'kernelfunction' is an exact function pointer
  checkCudaError(cudaFuncGetAttributes(&attr, (const void*)kernelfunction));

  //printf("[cudaLaunch] regs=%d, smem=%zu, maxThreadsPerBlock=%d, from=%s\n",
  //            attr.numRegs, (size_t)attr.sharedSizeBytes, attr.maxThreadsPerBlock, src.c_str());
  int BLOCKDIM_local;
  if (src != "strayfieldkernel.cu") {
    BLOCKDIM_local = BLOCKDIM;
  } else {
    // strayfieldkernel uses a lot of registers, so we reduce block size to
    // increase occupancy
    BLOCKDIM_local = 256;
  }
  dim3 blockDims(BLOCKDIM_local);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  //checkCudaError(cudaDeviceSynchronize());
}

template <typename... Arguments>
void cudaLaunchReductionKernel(void (*kernelfunction)(Arguments...),
                               Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims(1);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}
