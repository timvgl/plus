#include "cudalaunch.hpp"
#include "demagkernel.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "newell.hpp"

DemagKernel::DemagKernel(Grid grid, real3 cellsize)
    : cellsize_(cellsize), grid_(grid) {
  kernel_ = new Field(grid_, 6);
  compute();
}

DemagKernel::DemagKernel(Grid dst, Grid src, real3 cellsize)
    : cellsize_(cellsize), grid_(kernelGrid(dst, src)) {
  kernel_ = new Field(grid_, 6);
  compute();
}

DemagKernel::~DemagKernel() {
  delete kernel_;
}

__global__ void k_demagKernel(CuField kernel, real3 cellsize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!kernel.cellInGrid(idx))
    return;
  int3 coo = kernel.grid.index2coord(idx);
  kernel.setValueInCell(idx, 0, calcNewellNxx(coo, cellsize));
  kernel.setValueInCell(idx, 1, calcNewellNyy(coo, cellsize));
  kernel.setValueInCell(idx, 2, calcNewellNzz(coo, cellsize));
  kernel.setValueInCell(idx, 3, calcNewellNxy(coo, cellsize));
  kernel.setValueInCell(idx, 4, calcNewellNxz(coo, cellsize));
  kernel.setValueInCell(idx, 5, calcNewellNyz(coo, cellsize));
}

void DemagKernel::compute() {
  cudaLaunch(grid_.ncells(), k_demagKernel, kernel_->cu(), cellsize_);
}

Grid DemagKernel::grid() const {
  return grid_;
}
real3 DemagKernel::cellsize() const {
  return cellsize_;
}

const Field* DemagKernel::field() const {
  return kernel_;
}

Grid DemagKernel::kernelGrid(Grid dst, Grid src) {
  int3 size = src.size() + dst.size() - int3{1, 1, 1};

  // add padding to get even dimensions if size is larger than 5
  // this will make the fft on this grid mush more efficient
  if (size.x > 5 && size.x % 2 == 1)
    size.x += 1;
  if (size.y > 5 && size.y % 2 == 1)
    size.y += 1;
  if (size.z > 5 && size.z % 2 == 1)
    size.z += 1;

  int3 origin = src.origin() + src.size() - dst.origin() - size;
  return Grid(size, origin);
}