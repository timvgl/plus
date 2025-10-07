#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>

#define checkCudaError(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

#define checkCudaErrorMsg(ans, msg) \
  { gpuAssert((ans), __FILE__, __LINE__, true, (msg)); }

inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true,
                      std::string msg = "") {
  if (code != cudaSuccess) {
    if (!msg.empty()) {
      fprintf(stderr, "GPUassert: %s %s %d with %s\n", cudaGetErrorString(code), file,
            line, msg.c_str());
    }
    else {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    }
    
    if (abort)
      exit(code);
  }
}
