#include "cudastream.hpp"

cudaStream_t stream0;
cudaStream_t streamFFT;

cudaStream_t getCudaStream() {
  if (!stream0)
    cudaStreamCreate(&stream0);
  return stream0;
}

cudaStream_t getCudaStreamFFT() {
  if (!streamFFT)
    cudaStreamCreate(&streamFFT);
  return streamFFT;
}
