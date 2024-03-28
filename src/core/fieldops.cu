#include <stdexcept>
#include <vector>

#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"

__global__ void k_addFields(CuField y,
                            real a1,
                            const CuField x1,
                            real a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;
  for (int c = 0; c < y.ncomp; c++) {
    real term1 = a1 * x1.valueAt(idx, c % x1.ncomp);
    real term2 = a2 * x2.valueAt(idx, c % x2.ncomp);
    y.setValueInCell(idx, c, term1 + term2);
  }
}

__global__ void k_addFields(CuField y,
                            real3 a1,
                            const CuField x1,
                            real3 a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;

  real3 term1 = a1 * x1.FM_vectorAt(idx);
  real3 term2 = a2 * x2.FM_vectorAt(idx);
  y.setVectorInCell(idx, term1 + term2);
}

__global__ void k_addFields(CuField y,
                            real6 a1,
                            const CuField x1,
                            real6 a2,
                            const CuField x2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;

  real6 term1 = a1 * x1.AFM_vectorAt(idx);
  real6 term2 = a2 * x2.AFM_vectorAt(idx);
  y.setVectorInCell(idx, term1 + term2);
}

inline void add(Field& y, real a1, const Field& x1, real a2, const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems)");
  }
  if ((x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) ) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

inline void add(Field& y,
                real3 a1,
                const Field& x1,
                real3 a2,
                const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems)");
  }
  if (x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components");
  }
  if (x1.ncomp() != 3) {
    throw std::invalid_argument("Fields should have 3 components.");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

inline void add(Field& y,
                real6 a1,
                const Field& x1,
                real6 a2,
                const Field& x2) {
  if (x1.system() != y.system() || x2.system() != y.system()) {
    throw std::invalid_argument(
        "Fields can not be added together because they belong to different "
        "systems)");
  }
  if (x1.ncomp() != y.ncomp() || x1.ncomp() != y.ncomp()) {
    throw std::invalid_argument(
        "Fields can not be added because they do not have the same number of "
        "components");
  }
  if (x1.ncomp() != 6) {
    throw std::invalid_argument("Fields should have 6 components.");
  }
  int ncells = y.grid().ncells();
  cudaLaunch(ncells, k_addFields, y.cu(), a1, x1.cu(), a2, x2.cu());
}

Field add(real a1, const Field& x1, real a2, const Field& x2) {
  Field y(x1.system(), x1.ncomp());
  add(y, a1, x1, a2, x2);
  return y;
}

Field add(const Field& x1, const Field& x2) {
  return add(1, x1, 1, x2);
}

void addTo(Field& y, real a, const Field& x) {
  add(y, 1, y, a, x);
}

void addTo(Field& y, real3 a, const Field& x) {
  real3 a0 = real3{1, 1, 1};
  add(y, a0, y, a, x);
}

void addTo(Field& y, real6 a, const Field& x) {
  real6 a0 = real6{1, 1, 1, 1, 1, 1};
  add(y, a0, y, a, x);
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

__global__ void k_addConstant(CuField y,
                              const CuField x,
                              real value,
                              int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!y.cellInGeometry(idx))
    return;
  y.setValueInCell(idx, comp, x.valueAt(idx, comp) + value);
}

__global__ void k_normalize(CuField dst, const CuField src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dst.cellInGeometry(idx))
    return;

  int comp = src.ncomp;
  
  if (comp == 3) {
    real norm2 = 0.0;
    for (int c = 0; c < comp; c++) {
      real v = src.valueAt(idx, c);
      norm2 += v * v;
    }
    real invnorm = rsqrt(norm2);
    for (int c = 0; c < comp; c++) {
      real value = src.valueAt(idx, c) * invnorm;
      dst.setValueInCell(idx, c, value);
    }
  }
  else if (comp == 6) {
    real2 norm2 = real2{0., 0.};
    for (int c = 0; c < comp - 3; c++) {
      real v = src.valueAt(idx, c);
      real u = src.valueAt(idx, c + 3);
      norm2 += real2{v * v, u * u};
    }
    real2 invnorm = real2{rsqrt(norm2.x), rsqrt(norm2.y)};
    for (int c = 0; c < comp - 3; c++) {
      real vvalue = src.valueAt(idx, c) * invnorm.x;
      real uvalue = src.valueAt(idx, c + 3) * invnorm.y;
      dst.setValueInCell(idx, c, vvalue);
      dst.setValueInCell(idx, c + 3, uvalue);
    }
  }
}


Field normalized(const Field& src) {
  Field dst(Field(src.system(), src.ncomp()));
  cudaLaunch(dst.grid().ncells(), k_normalize, dst.cu(), src.cu());
  return dst;
}

void normalize(Field& f) {
  cudaLaunch(f.grid().ncells(), k_normalize, f.cu(), f.cu());
}

Field operator*(real3 a, const Field& x) {
  Field y(x.system(), x.ncomp());
  real3 a0 = real3{0, 0, 0};
  add(y, a0, x, a, x);
  return y;
}

Field operator*(real6 a, const Field& x) {
  Field y(x.system(), x.ncomp());
  real6 a0 = real6{0, 0, 0, 0, 0, 0};
  add(y, a0, x, a, x);
  return y;
}
