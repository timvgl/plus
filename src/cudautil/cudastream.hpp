#pragma once
#include "cudaerror.hpp"


cudaStream_t getCudaStream();
cudaStream_t getCudaStreamFFT();
cudaStream_t getCudaStreamGC();

inline void fenceStreamToStream(cudaStream_t src, cudaStream_t dst) {
  if (!src || src == dst) return;
  cudaEvent_t ev;
  checkCudaError(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  checkCudaError(cudaEventRecord(ev, src));
  checkCudaError(cudaStreamWaitEvent(dst, ev, 0));
  checkCudaError(cudaEventDestroy(ev));
}