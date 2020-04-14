#pragma once

#include "cudaerror.hpp"
#include "cudastream.hpp"

// Source: https://devblogs.nvidia.com/cplusplus-11-in-cuda-variadic-templates/
template <typename... Arguments>
void cudaLaunch(int N,
                void (*kernelfunction)(Arguments...),
                Arguments... args) {
  dim3 blockDims(512);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}