#include <cufft.h>

#include <memory>
#include <vector>

#include "cudalaunch.hpp"
#include "demagconvolution.hpp"
#include "demagkernel.hpp"
#include "field.hpp"

#define __CUDAOP__ inline __device__ __host__

__CUDAOP__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCadd(a, b);
}

__CUDAOP__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCmul(a, b);
}

__global__ static void k_pad(CuField out, CuField in) {
  if (!out.cellInGrid())
    return;
  int3 coo = out.grid.idx2coo(blockIdx.x * blockDim.x + threadIdx.x);
  int3 coo_ = coo - out.grid.origin() + in.grid.origin();
  for (int c = 0; c < out.ncomp; c++) {
    real value = in.cellInGrid(coo_) ? in.cellValue(coo_, c) : 0.0;
    out.setCellValue(c, value);
  }
}

__global__ static void k_unpad(CuField out, CuField in) {
  if (!out.cellInGrid())
    return;
  int3 coo = out.grid.idx2coo(blockIdx.x * blockDim.x + threadIdx.x);
  int3 coo_ = coo - out.grid.origin() + in.grid.origin() + in.grid.size() -
              out.grid.size();
  for (int c = 0; c < out.ncomp; c++) {
    out.setCellValue(c, in.cellValue(coo_, c));
  }
}

static void checkCufftResult(cufftResult result) {
  if (result != CUFFT_SUCCESS)
    throw std::runtime_error("cufft error in demag convolution");
}

__global__ static void k_apply_kernel(cufftDoubleComplex* hx,
                                      cufftDoubleComplex* hy,
                                      cufftDoubleComplex* hz,
                                      cufftDoubleComplex* mx,
                                      cufftDoubleComplex* my,
                                      cufftDoubleComplex* mz,
                                      cufftDoubleComplex* kxx,
                                      cufftDoubleComplex* kyy,
                                      cufftDoubleComplex* kzz,
                                      cufftDoubleComplex* kxy,
                                      cufftDoubleComplex* kxz,
                                      cufftDoubleComplex* kyz,
                                      cufftDoubleComplex preFactor,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  hx[i] = preFactor * (kxx[i] * mx[i] + kxy[i] * my[i] + kxz[i] * mz[i]);
  hy[i] = preFactor * (kxy[i] * mx[i] + kyy[i] * my[i] + kyz[i] * mz[i]);
  hz[i] = preFactor * (kxz[i] * mx[i] + kyz[i] * my[i] + kzz[i] * mz[i]);
}

DemagConvolution::DemagConvolution(Grid grid, real3 cellsize)
    : grid_(grid),
      cellsize_(cellsize),
      kernel_(grid, grid, cellsize),
      kfft(6),
      hfft(3),
      mfft(3) {
  int3 size = kernel_.grid().size();
  fftSize = {size.x / 2 + 1, size.y, size.z};
  int ncells = fftSize.x * fftSize.y * fftSize.z;

  for (auto& p : kfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));
  for (auto& p : mfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));
  for (auto& p : hfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));

  checkCufftResult(
      cufftPlan3d(&forwardPlan, size.z, size.y, size.x, CUFFT_D2Z));
  checkCufftResult(
      cufftPlan3d(&backwardPlan, size.z, size.y, size.x, CUFFT_Z2D));

  for (int comp = 0; comp < 6; comp++)
    checkCufftResult(cufftExecD2Z(forwardPlan, kernel_.field()->devptr(comp),
                                  kfft.at(comp)));
}

DemagConvolution::~DemagConvolution() {
  for (auto p : mfft)
    cudaFree(p);
  for (auto p : kfft)
    cudaFree(p);
  for (auto p : hfft)
    cudaFree(p);

  checkCufftResult(cufftDestroy(forwardPlan));
  checkCufftResult(cufftDestroy(backwardPlan));
}

void DemagConvolution::exec(Field* h, const Field* m, real msat) const {
  // add padding
  std::unique_ptr<Field> mpad(new Field(kernel_.grid(), 3));
  cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m->cu());

  // Forward fourier transforms
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
        cufftExecD2Z(forwardPlan, mpad->devptr(comp), mfft.at(comp)));

  // apply kernel on m_fft
  int ncells = fftSize.x * fftSize.y * fftSize.z;
  const real MU0 = 4 * M_PI * 1e-7;  // TODO: move this to a general place
  cufftDoubleComplex preFactor =
      make_cuDoubleComplex(-MU0 * msat / kernel_.grid().ncells(), 0);
  cudaLaunch(ncells, k_apply_kernel, hfft.at(0), hfft.at(1), hfft.at(2),
             mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1),
             kfft.at(2), kfft.at(3), kfft.at(4), kfft.at(5), preFactor, ncells);

  // backward fourier transfrom
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
        cufftExecZ2D(backwardPlan, hfft.at(comp), mpad->devptr(comp)));

  // unpad
  cudaLaunch(h->grid().ncells(), k_unpad, h->cu(), mpad->cu());
}