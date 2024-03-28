#pragma once

#include <cuComplex.h>
#include <cuda_runtime_api.h>

#include <iostream>

#define SINGLE 1
#define DOUBLE 2

#ifndef FP_PRECISION
#define FP_PRECISION SINGLE
#endif

#if FP_PRECISION == SINGLE
using real = float;
using real2 = float2;
using real3 = float3;
using real6 = float6;
using complex = cuComplex;
#elif FP_PRECISION == DOUBLE
using real = double;
using real2 = double2;
using real3 = double3;
using real6 = double6;
using complex = cuDoubleComplex;
#else
#error FP_PRECISION should be SINGLE or DOUBLE
#endif

#define __CUDAOP__ inline __device__ __host__

__CUDAOP__ void operator+=(int3& a, const int3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator+=(int6& a, const int6& b) {
  a.x1 += b.x1;
  a.y1 += b.y1;
  a.z1 += b.z1;
  a.x2 += b.x2;
  a.y2 += b.y2;
  a.z2 += b.z2;
}

__CUDAOP__ void operator-=(int3& a, const int3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ void operator-=(int6& a, const int6& b) {
  a.x1 -= b.x1;
  a.y1 -= b.y1;
  a.z1 -= b.z1;
  a.x2 -= b.x2;
  a.y2 -= b.y2;
  a.z2 -= b.z2;
}

__CUDAOP__ int3 operator+(const int3& a, const int3& b) {
  return int3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ int6 operator+(const int6& a, const int6& b) {
  return int6{a.x1 + b.x1, a.y1 + b.y1, a.z1 + b.z1,
              a.x2 + b.x2, a.y2 + b.y2, a.z2 + b.z2};
}

__CUDAOP__ int3 operator-(const int3& a) {
  return int3{-a.x, -a.y, -a.z};
}

__CUDAOP__ int6 operator-(const int6& a) {
  return int6{-a.x1, -a.y1, -a.z1, -a.x2, -a.y2, -a.z2};
}

__CUDAOP__ int3 operator-(const int3& a, const int3& b) {
  return int3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ int6 operator-(const int6& a, const int6& b) {
  return int6{a.x1 - b.x1, a.y1 - b.y1, a.z1 - b.z1,
              a.x2 - b.x2, a.y2 - b.y2, a.z2 - b.z2};
}

__CUDAOP__ int3 operator*(const int3& a, const int3& b) {
  return int3{a.x * b.x, a.y * b.y, a.z * b.z};
}

__CUDAOP__ int6 operator*(const int6& a, const int6& b) {
  return int6{a.x1 * b.x1, a.y1 * b.y1, a.z1 * b.z1,
              a.x2 * b.x2, a.y2 * b.y2, a.z2 * b.z2};
}

__CUDAOP__ int3 operator/(const int3& a, const int3& b) {
  return int3{a.x / b.x, a.y / b.y, a.z / b.z};
}

__CUDAOP__ int6 operator/(const int6& a, const int6& b) {
  return int6{a.x1 / b.x1, a.y1 / b.y1, a.z1 / b.z1,
              a.x2 / b.x2, a.y2 / b.y2, a.z2 / b.z2};
}

__CUDAOP__ bool operator==(const int3& a, const int3& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__CUDAOP__ bool operator==(const int6& a, const int6& b) {
  return (a.x1 == b.x1) && (a.y1 == b.y1) && (a.z1 == b.z1)
      && (a.x2 == b.x2) && (a.y2 == b.y2) && (a.z1 == b.z2);
}

__CUDAOP__ bool operator!=(const int3& a, const int3& b) {
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

__CUDAOP__ bool operator!=(const int6& a, const int6& b) {
  return (a.x1 != b.x1) || (a.y1 != b.y1) || (a.z1 != b.z1)
      || (a.x2 != b.x2) || (a.y2 != b.y2) || (a.z2 != b.z2);
}

__CUDAOP__ bool operator!=(const real2& a, const real2& b) {
  return (a.x != b.x) || (a.y != b.y);
}

inline __host__ std::ostream& operator<<(std::ostream& os, const int3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}

inline __host__ std::ostream& operator<<(std::ostream& os, const int6 a) {
  os << "(" << a.x1 << "," << a.y1 << "," << a.z1 << 
        "," << a.x2 << "," << a.y2 << "," << a.z2 << ")";
  return os;
}

__CUDAOP__ void operator+=(real3& a, const real3& b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__CUDAOP__ void operator+=(real6& a, const real6& b) {
  a.x1 += b.x1;
  a.y1 += b.y1;
  a.z1 += b.z1;
  a.x2 += b.x2;
  a.y2 += b.y2;
  a.z2 += b.z2;
}

__CUDAOP__ void operator +=(real2& a, const real2& b) {
  a.x += b.x;
  a.y += b.y;
}

__CUDAOP__ void operator+=(real3& a, const real& b) {
  a.x += b;
  a.y += b;
  a.z += b;
}

__CUDAOP__ void operator+=(real6& a, const real& b) {
  a.x1 += b;
  a.y1 += b;
  a.z1 += b;
  a.x2 += b;
  a.y2 += b;
  a.z2 += b;
}

__CUDAOP__ void operator-=(real3& a, const real3& b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
}

__CUDAOP__ void operator-=(real6& a, const real6& b) {
  a.x1 -= b.x1;
  a.y1 -= b.y1;
  a.z1 -= b.z1;
  a.x2 -= b.x2;
  a.y2 -= b.y2;
  a.z2 -= b.z2;
}

__CUDAOP__ void operator-=(real2& a, const real2& b) {
  a.x -= b.x;
  a.y -= b.y;
}

__CUDAOP__ void operator-=(real3& a, const real& b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
}

__CUDAOP__ void operator-=(real6& a, const real& b) {
  a.x1 -= b;
  a.y1 -= b;
  a.z1 -= b;
  a.x2 -= b;
  a.y2 -= b;
  a.z2 -= b;
}

__CUDAOP__ void operator*=(real3& a, const real& b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
}

__CUDAOP__ void operator*=(real6& a, const real& b) {
  a.x1 *= b;
  a.y1 *= b;
  a.z1 *= b;
  a.x2 *= b;
  a.y2 *= b;
  a.z2 *= b;
}

__CUDAOP__ void operator*=(real6& a, const real2& b) {
  a.x1 *= b.x;
  a.y1 *= b.x;
  a.z1 *= b.x;
  a.x2 *= b.y;
  a.y2 *= b.y;
  a.z2 *= b.y;
}

__CUDAOP__ void operator/=(real3& a, const real& b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
}

__CUDAOP__ void operator/=(real6& a, const real& b) {
  a.x1 /= b;
  a.y1 /= b;
  a.z1 /= b;
  a.x2 /= b;
  a.y2 /= b;
  a.z2 /= b;
}

__CUDAOP__ void operator/=(real6& a, const real2& b) {
  a.x1 /= b.x;
  a.y1 /= b.x;
  a.z1 /= b.x;
  a.x2 /= b.y;
  a.y2 /= b.y;
  a.z2 /= b.y;
}

__CUDAOP__ real3 operator+(const real3& a, const real3& b) {
  return real3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__CUDAOP__ real6 operator+(const real6& a, const real6& b) {
    return real6{a.x1 + b.x1, a.y1 + b.y1, a.z1 + b.z1,
                 a.x2 + b.x2, a.y2 + b.y2, a.z2 + b.z2};
}

__CUDAOP__ real3 operator+(const real& a, const real3& b) {
  return real3{a + b.x, a + b.y, a + b.z};
}

__CUDAOP__ real6 operator+(const real& a, const real6& b) {
  return real6{a + b.x1, a + b.y1, a + b.z1,
               a + b.x2, a + b.y2, a + b.z2};
}

__CUDAOP__ real3 operator+(const real3& a, const real& b) {
  return real3{a.x + b, a.y + b, a.z + b};
}

__CUDAOP__ real6 operator+(const real6& a, const real& b) {
  return real6{a.x1 + b, a.y1 + b, a.z1 + b,
               a.x2 + b, a.y2 + b, a.z2 + b};
}

__CUDAOP__ real2 operator+(const real2& a, const real2& b) {
  return real2{a.x + b.x, a.y + b.y};
}

__CUDAOP__ real2 operator+(const real& a, const real2& b) {
  return real2{a + b.x, a + b.y};
}

__CUDAOP__ real2 operator+(const real2& a, const real& b) {
  return real2{a.x + b, a.y + b};
}

__CUDAOP__ real3 operator-(const real3& a) {
  return real3{-a.x, -a.y, -a.z};
}

__CUDAOP__ real6 operator-(const real6& a) {
  return real6{-a.x1, -a.y1, -a.z1, -a.x2, -a.y2, -a.z2};
}

__CUDAOP__ real2 operator-(const real2& a) {
  return real2{-a.x, -a.y};
}

__CUDAOP__ real3 operator-(const real3& a, const real3& b) {
  return real3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__CUDAOP__ real6 operator-(const real6& a, const real6& b) {
  return real6{a.x1 - b.x1, a.y1 - b.y1, a.z1 - b.z1,
               a.x2 - b.x2, a.y2 - b.y2, a.z2 - b.z2};
}

__CUDAOP__ real2 operator-(const real2& a, const real2& b) {
  return real2{a.x - b.x, a.y - b.y};
}

__CUDAOP__ real3 operator-(const real3& a, const real& b) {
  return real3{a.x - b, a.y - b, a.z - b};
}

__CUDAOP__ real6 operator-(const real6& a, const real& b) {
  return real6{a.x1 - b, a.y1 - b, a.z1 - b,
               a.x2 - b, a.y2 - b, a.z2 - b};
}

__CUDAOP__ real3 operator-(const real& a, const real3& b) {
  return real3{a - b.x, a - b.y, a - b.z};
}

__CUDAOP__ real6 operator-(const real& a, const real6& b) {
  return real6{a - b.x1, a - b.y1, a - b.z1,
               a - b.x2, a - b.y2, a - b.z2};
}

__CUDAOP__ real2 operator-(const real& a, const real2& b) {
  return real2{a - b.x, a - b.y};
}

__CUDAOP__ real3 operator*(const real& a, const real3& b) {
  return real3{a * b.x, a * b.y, a * b.z};
}

__CUDAOP__ real6 operator*(const real& a, const real6& b) {
  return real6{a * b.x1, a * b.y1, a * b.z1,
               a * b.x2, a * b.y2, a * b.z2};
}

__CUDAOP__ real2 operator*(const real& a, const real2& b) {
  return real2{a * b.x, a * b.y};
}

__CUDAOP__ real2 operator*(const real2& a, const real& b) {
  return real2{a.x * b, a.y * b};
}

__CUDAOP__ real6 operator*(const real2& a, const real6& b) {
  return real6{a.x * b.x1, a.x * b.y1, a.x * b.z1,
               a.y * b.x2, a.y * b.y2, a.y * b.z2};
}

__CUDAOP__ real3 operator*(const real3& a, const real& b) {
  return real3{a.x * b, a.y * b, a.z * b};
}

__CUDAOP__ real6 operator*(const real6& a, const real& b) {
  return real6{a.x1 * b, a.y1 * b, a.z1 * b,
               a.x2 * b, a.y2 * b, a.z2 * b};
}

__CUDAOP__ real6 operator*(const real3& a, const real6& b) {
  return real6{a.x * b.x1, a.y * b.y1, a.z * b.z1,
               a.x * b.x2, a.y * b.y2, a.z * b.z2};
}

__CUDAOP__ real6 operator*(const real6& a, const real2& b) {
  return real6{a.x1 * b.x, a.y1 * b.x, a.z1 * b.x,
               a.x2 * b.y, a.y2 * b.y, a.z2 * b.y};
}

__CUDAOP__ real3 operator*(const real3& a, const real3& b) {
  return real3{a.x * b.x, a.y * b.y, a.z * b.z};
}

__CUDAOP__ real6 operator*(const real6& a, const real6& b) {
  return real6{a.x1 * b.x1, a.y1 * b.y1, a.z1 * b.z1,
               a.x2 * b.x2, a.y2 * b.y2, a.z2 * b.z2};
}

__CUDAOP__ real2 operator*(const real2& a, const real2& b) {
  return real2{a.x * b.x, a.y * b.y};
}

__CUDAOP__ real3 operator/(const real3& a, const real& b) {
  return real3{a.x / b, a.y / b, a.z / b};
}

__CUDAOP__ real6 operator/(const real6& a, const real& b) {
  return real6{a.x1 / b, a.y1 / b, a.z1 / b,
               a.x2 / b, a.y2 / b, a.z2 / b};
}

__CUDAOP__ real3 operator/(const real3& a, const real3& b) {
  return real3{a.x / b.x, a.y / b.y, a.z / b.z};
}

__CUDAOP__ real6 operator/(const real6& a, const real6& b) {
    return real6{a.x1 / b.x1, a.y1 / b.y1, a.z1 / b.z1,
                 a.x2 / b.x2, a.y2 / b.y2, a.z2 / b.z2};
}

__CUDAOP__ real6 operator/(const real6& a, const real2& b) {
    return real6{a.x1 / b.x, a.y1 / b.x, a.z1 / b.x,
                 a.x2 / b.y, a.y2 / b.y, a.z2 / b.y};
}

__CUDAOP__ real2 operator/(const real& a, const real2& b) {
  return real2{a / b.x, a / b.y};
}

__CUDAOP__ real2 operator/(const real2& a, const real& b) {
  return real2{a.x / b, a.y / b};
}

__CUDAOP__ real dot(const real3& a, const real3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__CUDAOP__ real2 dot(const real6& a, const real6& b) {
  return real2{a.x1 * b.x1 + a.y1 * b.y1 + a.z1 * b.z1,
               a.x2 * b.x2 + a.y2 * b.y2 + a.z2 * b.z2};
}

__CUDAOP__ real2 dot(const real6& a, const real3& b) {
  return real2{a.x1 * b.x + a.y1 * b.y + a.z1 * b.z,
               a.x2 * b.x + a.y2 * b.y + a.z2 * b.z};
}

__CUDAOP__ real3 cross(const real3& a, const real3& b) {
  return real3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x};
}

__CUDAOP__ real6 cross(const real6& a, const real6& b) {
  return real6{a.y1 * b.z1 - a.z1 * b.y1, a.z1 * b.x1 - a.x1 * b.z1,
               a.x1 * b.y1 - a.y1 * b.x1,
               a.y2 * b.z2 - a.z2 * b.y2, a.z2 * b.x2 - a.x2 * b.z2,
               a.x2 * b.y2 - a.y2 * b.x2};
}

__CUDAOP__ real6 cross(const real3& a, const real6& b) {
  real3 b1 = {b.x1, b.y1, b.z1};
  real3 b2 = {b.x2, b.y2, b.z2};
  real3 p1 = cross(a, b1);
  real3 p2 = cross(a, b2);

  return real6{p1.x, p1.y, p1.z, p2.x, p2.y, p2.z};
}

__CUDAOP__ real6 cross(const real6& a, const real3& b) {
  real3 a1 = {a.x1, a.y1, a.z1};
  real3 a2 = {a.x2, a.y2, a.z2};
  real3 p1 = cross(a1, b);
  real3 p2 = cross(a2, b);

  return real6{p1.x, p1.y, p1.z, p2.x, p2.y, p2.z};
}

__CUDAOP__ real2 sqrt(const real2& a) {
  return real2{sqrt(a.x), sqrt(a.y)};
}

__CUDAOP__ real norm(const real3& a) {
  return sqrt(dot(a, a));
}

__CUDAOP__ real2 norm(const real6& a) {
  return sqrt(dot(a, a));
}

__CUDAOP__ bool operator==(const real2& a, const real2& b) {
  return (a.x == b.x) && (a.y == b.y);
}

// Returns the normalized vector.
// If the norm is zero, it returns a zero vector
__CUDAOP__ real3 normalized(const real3& a) {
  real scalingsFactor = (norm(a) == real(0)) ? 0. : 1. / norm(a);
  return scalingsFactor * a;
}

__CUDAOP__ real6 normalized(const real6& a) {
  real2 scalingsFactor = (norm(a) == real2{0., 0.}) ? real2{0.,0.} : 1. / norm(a);
  return scalingsFactor * a;
}

__CUDAOP__ bool operator==(const real3& a, const real3& b) {
  return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

__CUDAOP__ bool operator==(const real6& a, const real6& b) {
  return (a.x1 == b.x1) && (a.y1 == b.y1) && (a.z1 == b.z1)
      && (a.x2 == b.x2) && (a.y2 == b.y2) && (a.z2 == b.z2);
}

__CUDAOP__ bool operator!=(const real3& a, const real3& b) {
  return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);
}

__CUDAOP__ bool operator!=(const real6& a, const real6& b) {
  return (a.x1 != b.x1) || (a.y1 != b.y1) || (a.z1 != b.z1)
      || (a.x2 != b.x2) || (a.y2 != b.y2) || (a.z2 != b.z2);
}

inline __host__ std::ostream& operator<<(std::ostream& os, const real3 a) {
  os << "(" << a.x << "," << a.y << "," << a.z << ")";
  return os;
}

inline __host__ std::ostream& operator<<(std::ostream& os, const real6 a) {
  os << "(" << a.x1 << "," << a.y1 << "," << a.z1 << "), "
     << "(" << a.x2 << "," << a.y2 << "," << a.z2 << ")";
  return os;
}
