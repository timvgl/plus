#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"

__global__ void k_addFields(CuField y,
                            real a1,
                            CuField x1,
                            real a2,
                            CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGrid(idx))
    return;
  for (int c = -0; c < y.ncomp; c++) {
    real term1 = a1 * x1.valueAt(idx, c);
    real term2 = a2 * x2.valueAt(idx, c);
    y.setValueInCell(idx, c, term1 + term2);
  }
}

// TODO: throw error if grids or number of components do not match
void add(Field* y, real a1, const Field* x1, real a2, const Field* x2) {
  int ncells = y->grid().ncells();
  cudaLaunch(ncells, k_addFields, y->cu(), a1, x1->cu(), a2, x2->cu());
}

void add(Field* y, const Field* x1, const Field* x2) {
  add(y, 1, x1, 1, x2);
}

// TODO: this can be done much more efficient
void add(Field* y, std::vector<const Field*> x, std::vector<real> weights) {
  // TODO:: throw error if inputs are not compatible
  if (x.size() == 1) {
    add(y, 0, x.at(0), weights.at(0), x.at(0));
  }

  add(y, weights.at(0), x.at(0), weights.at(1), x.at(1));
  for (int n = 2; n < x.size(); n++) {
    if (weights.at(n) != 0.0) {
      add(y, 1, y, weights.at(n), x.at(n));
    }
  }
}

__global__ void k_addConstant(CuField y, CuField x, real value, int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGrid(idx))
    return;
  y.setValueInCell(idx, comp, x.valueAt(idx, comp)+value );
}

void addConstant(Field *y, Field *x, real value, int comp) {
  int ncells = y->grid().ncells();
  cudaLaunch(ncells, k_addConstant, y->cu(), x->cu(), value, comp);
}

__global__ void k_normalize(CuField dst, CuField src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dst.cellInGrid(idx))
    return;
  real norm2 = 0.0;
  for (int c = 0; c < src.ncomp; c++) {
    real v = src.valueAt(idx, c);
    norm2 += v * v;
  }
  real invnorm = rsqrt(norm2);
  for (int c = 0; c < src.ncomp; c++) {
    real value = src.valueAt(idx,c) * invnorm;
    dst.setValueInCell(idx, c, value);
  }
}

void normalized(Field* dst, const Field* src) {
  // TODO: check field dimensions
  cudaLaunch(dst->grid().ncells(), k_normalize, dst->cu(), src->cu());
}

std::unique_ptr<Field> normalized(const Field* src) {
  std::unique_ptr<Field> dst(new Field(src->grid(), src->ncomp()));
  normalized(dst.get(), src);
  return dst;
}

void normalize(Field* f) {
  normalized(f, f);
}