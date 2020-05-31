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

Field add(real a1, const Field& x1, real a2, const Field& x2) {
  // TODO: check grid sizes and number of components
  Field y(x1.grid(), x1.ncomp());
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
  return y;
}

Field add(const Field& x1, const Field& x2) {
  return add(1, x1, 1, x2);
}

void addTo(Field& y, real a, const Field& x) {
  // TODO: check grid sizes and number of components
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), real(1), y.cu(), a, x.cu());
}

// TODO: this can be done much more efficient
Field add(std::vector<const Field*> x, std::vector<real> weights) {
  // TODO:: throw error if inputs are not compatible
  Field y = weights.at(0) * (*x.at(0));
  if (x.size() == 1) {
    return y;
  }

  for (int n = 1; n < x.size(); n++) {
    if (weights.at(n) != 0.0) {
      addTo(y, weights.at(n), *x.at(n));
    }
  }
  return y;
}

Field operator*(real a, const Field& x) {
  return add(0, x, a, x);
}

__global__ void k_addConstant(CuField y, CuField x, real value, int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGrid(idx))
    return;
  y.setValueInCell(idx, comp, x.valueAt(idx, comp) + value);
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
    real value = src.valueAt(idx, c) * invnorm;
    dst.setValueInCell(idx, c, value);
  }
}

Field normalized(const Field& src) {
  Field dst(Field(src.grid(), src.ncomp()));
  cudaLaunch(dst.grid().ncells(), k_normalize, dst.cu(), src.cu());
  return dst;
}

void normalize(Field& f) {
  cudaLaunch(f.grid().ncells(), k_normalize, f.cu(), f.cu());
}