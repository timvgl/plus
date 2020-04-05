#include <cufft.h>

#include <memory>
#include <vector>

#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "math.h"
#include "world.hpp"

DemagField::DemagField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "demag_field", "T"),
      demagkernel_(ferromagnet->grid(),
                   ferromagnet->grid(),
                   ferromagnet->world()->cellsize()) {}

#define __CUDAOP__ inline __device__ __host__

__CUDAOP__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCadd(a, b);
}

__CUDAOP__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) {
  return cuCmul(a, b);
}

__global__ static void k_move(CuField out, CuField in) {
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
    std::cout << "whoops" << std::endl;
}

static void move(Field* out, const Field* in) {
  cudaLaunch(out->grid().ncells(), k_move, out->cu(), in->cu());
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

void convolution(Field* out, const Field* m, const Field* kern, real msat) {
  Grid grid = kern->grid();

  // put m on kernel grid
  std::unique_ptr<Field> mpad(new Field(grid, 3));
  move(mpad.get(), m);  // pad

  int3 size = grid.size();
  int3 fftSize{size.x / 2 + 1, size.y, size.z};

  // allocate temporary gpu buffers
  std::vector<cufftDoubleComplex*> mfft(3);
  std::vector<cufftDoubleComplex*> hfft(3);
  std::vector<cufftDoubleComplex*> kfft(6);
  int ncells = fftSize.x * fftSize.y * fftSize.z;
  for (auto& p : mfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));
  for (auto& p : hfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));
  for (auto& p : kfft)
    cudaMalloc((void**)&p, ncells * sizeof(cufftDoubleComplex));

  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
  checkCufftResult(
      cufftPlan3d(&forwardPlan, size.z, size.y, size.x, CUFFT_D2Z));
  checkCufftResult(
      cufftPlan3d(&backwardPlan, size.z, size.y, size.x, CUFFT_Z2D));

  // Forward fourier transforms
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
        cufftExecD2Z(forwardPlan, mpad->devptr(comp), mfft.at(comp)));
  for (int comp = 0; comp < 6; comp++)
    checkCufftResult(
        cufftExecD2Z(forwardPlan, kern->devptr(comp), kfft.at(comp)));

  // apply kernel on m_fft
  const real MU0 = 4 * M_PI * 1e-7;  // TODO: move this to a general place
  cufftDoubleComplex preFactor =
      make_cuDoubleComplex(-MU0 * msat / grid.ncells(), 0);
  cudaLaunch(ncells, k_apply_kernel, hfft.at(0), hfft.at(1), hfft.at(2),
             mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1),
             kfft.at(2), kfft.at(3), kfft.at(4), kfft.at(5), preFactor, ncells);

  // backward fourier transfrom
  for (int comp = 0; comp < 3; comp++)
    checkCufftResult(
        cufftExecZ2D(backwardPlan, hfft.at(comp), mpad->devptr(comp)));

  // clean up temporary gpu memory buffers
  for (auto p : mfft)
    cudaFree(p);
  for (auto p : kfft)
    cudaFree(p);
  for (auto p : hfft)
    cudaFree(p);
  checkCufftResult(cufftDestroy(forwardPlan));
  checkCufftResult(cufftDestroy(backwardPlan));

  cudaLaunch(out->grid().ncells(), k_unpad, out->cu(), mpad->cu());  // unpad
}
__global__ void k_demagfield(CuField hField,
                             CuField mField,
                             CuField kernel,
                             real msat) {
  if (!hField.cellInGrid())
    return;

  real3 h{0, 0, 0};

  Grid g = mField.grid;
  int3 dstcoo = g.idx2coo(blockIdx.x * blockDim.x + threadIdx.x);

  for (int i = 0; i < g.ncells(); i++) {
    int3 srccoo = g.idx2coo(i);
    int3 dist = dstcoo - srccoo;

    real3 m = mField.cellVector(i);

    real nxx = kernel.cellValue(dist, 0);
    real nyy = kernel.cellValue(dist, 1);
    real nzz = kernel.cellValue(dist, 2);
    real nxy = kernel.cellValue(dist, 3);
    real nxz = kernel.cellValue(dist, 4);
    real nyz = kernel.cellValue(dist, 5);

    h.x -= nxx * m.x + nxy * m.y + nxz * m.z;
    h.y -= nxy * m.x + nyy * m.y + nyz * m.z;
    h.z -= nxz * m.x + nyz * m.y + nzz * m.z;
  }
  const real MU0 = 4 * M_PI * 1e-7;  // TODO: move this to a general place
  hField.setCellVector(msat * MU0 * h);
}

void DemagField::evalIn(Field* result) const {
  const Field* m = ferromagnet_->magnetization()->field();
  const Field* kernel = demagkernel_.field();
  real msat = ferromagnet_->msat;
  int ncells = result->grid().ncells();

  convolution(result, m, kernel, msat);

  //// brute method
  // cudaLaunch(ncells, k_demagfield, result->cu(), m->cu(), kernel->cu(),
  // msat);
}