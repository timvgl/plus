#include "cudalaunch.hpp"
#include "demagkernel.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "newell.hpp"

DemagKernel::DemagKernel(Grid dst, Grid src, real3 cellsize)
    : dstGrid_(dst),
      srcGrid_(src),
      cellsize_(cellsize),
      grid_(kernelGrid(dst, src)) {
  kernel_ = new Field(grid_, 6);
  compute();
}

DemagKernel::~DemagKernel() {
  delete kernel_;
}

__global__ void k_demagKernel(CuField* kernel, real3 cellsize) {
  if (!kernel->cellInGrid())
    return;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int3 coo = kernel->grid().idx2coo(idx);
  kernel->setCellValue(0, calcNewellNxx(coo, cellsize));
  kernel->setCellValue(1, calcNewellNyy(coo, cellsize));
  kernel->setCellValue(2, calcNewellNzz(coo, cellsize));
  kernel->setCellValue(3, calcNewellNxy(coo, cellsize));
  kernel->setCellValue(4, calcNewellNxz(coo, cellsize));
  kernel->setCellValue(5, calcNewellNyz(coo, cellsize));
}

void DemagKernel::compute() {
  cudaLaunch(grid_.ncells(), k_demagKernel, kernel_->cu(), cellsize_);
}

const Field* DemagKernel::field() const {
  return kernel_;
}

Grid DemagKernel::kernelGrid(Grid dst, Grid src) {
  int3 size = src.size() + dst.size() - int3{1, 1, 1};
  int3 origin = src.origin() - (dst.origin() + dst.size() - int3{1, 1, 1});
  return Grid(size, origin);
}