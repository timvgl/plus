#pragma once

#include "cudaerror.hpp"

// Source: https://devblogs.nvidia.com/cplusplus-11-in-cuda-variadic-templates/
template <typename... Arguments>
void cudaLaunch(int N,
                void (*kernelfunction)(Arguments...),
                Arguments... args) {
  dim3 blockDims(512);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  kernelfunction<<<gridDims, blockDims>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}