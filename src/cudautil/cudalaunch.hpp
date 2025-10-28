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
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr,
            "CUDA launch error after %s: %s (grid=%u, block=%u)\n",
            src.c_str(), cudaGetErrorString(err), gridDims.x, blockDims.x);
    abort();
  }
  //checkCudaError(cudaDeviceSynchronize());
}

template <typename... Arguments>
void cudaLaunchOn(cudaStream_t s0, std::string src, int N,
                void (*kernelfunction)(Arguments...),
                Arguments... args) {

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
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  //checkCudaError(cudaDeviceSynchronize());
}

template <typename... Arguments>
void cudaLaunchStrayfieldKernel(int N,
                void (*kernelfunction)(Arguments...),
                Arguments... args) {
  dim3 blockDims(BLOCKDIM / 2);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}

template <typename... Arguments>
void cudaLaunchReductionKernel(void (*kernelfunction)(Arguments...),
                               Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims(1);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaStreamSynchronize(s0));
}
