#include <stdexcept>

#include "cudalaunch.hpp"
#include "linsystem.hpp"
#include "vec.hpp"

namespace {
  struct LinSysCleanup {
    GVec b;
    GVec val;
    GpuBuffer<int> idx;
    cudaEvent_t lastUseEvent = nullptr;
  };

  static void CUDART_CB linsys_cleanup_cb(void* p) noexcept {
    auto* cap = static_cast<LinSysCleanup*>(p);
    if (cap->lastUseEvent) {
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap; // zerstört b/val/idx => Device-Buffer werden jetzt freigegeben
  }
}


LinearSystem::LinearSystem(int nrows, int maxNonZerosInRow)
    : nnz_(maxNonZerosInRow),
      b_(maxNonZerosInRow * nrows),
      matrixval_(GVec(maxNonZerosInRow * nrows)),
      matrixidx_(GpuBuffer<int>(maxNonZerosInRow * nrows)) {}

LinearSystem::~LinearSystem() {
  const bool nothing_to_free = (b_.size() == 0 && matrixval_.size() == 0 && matrixidx_.size() == 0);
  if (nothing_to_free) {
    if (lastUseEvent_) {
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    return;
  }

  auto* cap = new (std::nothrow) LinSysCleanup{};
  if (!cap) {
    // Fallback: blockierend und sicher
    if (lastUseEvent_) {
      cudaEventSynchronize(lastUseEvent_);
      cudaEventDestroy(lastUseEvent_);
      lastUseEvent_ = nullptr;
    }
    // Zerstörung der Member synchron:
    // (RAII von GVec/GpuBuffer kümmert sich ums Freigeben)
    return;
  }

  cap->b   = std::move(b_);
  cap->val = std::move(matrixval_);
  cap->idx = std::move(matrixidx_);
  cap->lastUseEvent = lastUseEvent_;
  lastUseEvent_ = nullptr;

  cudaStream_t s_gc = getCudaStreamGC();
  if (cap->lastUseEvent) {
    checkCudaError(cudaStreamWaitEvent(s_gc, cap->lastUseEvent, 0));
  }
  cudaError_t st = cudaLaunchHostFunc(s_gc, linsys_cleanup_cb, cap);
  if (st != cudaSuccess) {
    // Fallback: blockierend
    if (cap->lastUseEvent) {
      cudaEventSynchronize(cap->lastUseEvent);
      cudaEventDestroy(cap->lastUseEvent);
      cap->lastUseEvent = nullptr;
    }
    delete cap; // gibt die Member sofort frei
  }
}

void LinearSystem::markLastUse() {
  // versuche einen sinnvollen Stream zu nehmen
  cudaStream_t s = matrixval_.getStream();
  if (!s) s = b_.getStream();
  if (!s) s = getCudaStreamGC();

  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}

void LinearSystem::markLastUse(cudaStream_t s) {
  if (!lastUseEvent_) {
    checkCudaError(cudaEventCreateWithFlags(&lastUseEvent_, cudaEventDisableTiming));
  }
  checkCudaError(cudaEventRecord(lastUseEvent_, s));
}


// For a linear system Ax=b, this kernel computes y= ka * A*x + kb * b
__global__ static void k_apply(lsReal* y,
                               LinearSystem::CuData linsys,
                               lsReal* x,
                               lsReal ka,
                               lsReal kb) {
  const int rowidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rowidx >= linsys.nrows)
    return;

  lsReal ax = 0.0;  // A*x accumulator for this row

  for (int i = 0; i < linsys.nnz; i++) {
    const int colidx = linsys.matrixIdx(rowidx, i);
    if (colidx >= 0)
      ax += linsys.matrixVal(rowidx, i) * x[colidx];
  }
  y[rowidx] = ka * ax + kb * linsys.b[rowidx];
}

// For a linear system Ax=b, this function returns y= ka * A*x + kb * b
static GVec apply(const LinearSystem& sys,
                  const GVec& x,
                  lsReal ka,
                  lsReal kb) {
  if ((int)x.size() != sys.nRows()) {
    throw std::invalid_argument(
        "The numbers of rows in the linear system does not match the number of "
        "cells of the field");
  }
  GVec y(x.size());
  cudaLaunch("linsystem.cu", x.size(), k_apply, y.get(), sys.cu(), x.get(), ka, kb);
  const_cast<LinearSystem&>(sys).markLastUse();
  return y;
}

GVec LinearSystem::matrixmul(const GVec& x) const {
  return apply(*this, x, 1.0, 0.0);  // A*x
}

GVec LinearSystem::residual(const GVec& x) const {
  return apply(*this, x, -1.0, 1.0);  // b - A*x
}

LinearSystem::CuData LinearSystem::cu() const {
  return CuData{nRows(), nnz_, matrixidx_.get(), matrixval_.get(), b_.get()};
}
