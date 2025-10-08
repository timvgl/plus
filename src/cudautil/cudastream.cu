#include "cudastream.hpp"
#include "cudaerror.hpp"

cudaStream_t stream0;
cudaStream_t streamFFT;
cudaStream_t GCStream;

cudaStream_t getCudaStream() {
  if (!stream0)
    checkCudaError(cudaStreamCreate(&stream0));
  return stream0;
}

cudaStream_t getCudaStreamFFT() {
  if (!streamFFT)
    checkCudaError(cudaStreamCreate(&streamFFT));
  return streamFFT;
}

cudaStream_t getCudaStreamGC() {
  if (!GCStream)
    checkCudaError(cudaStreamCreateWithFlags(&GCStream, cudaStreamNonBlocking));
  return GCStream;
}
