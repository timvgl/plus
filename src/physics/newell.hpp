#pragma once

#include "datatypes.hpp"

// The Newell functions as derived by Newell in doi.org/10.1029/93JB00694
// In the newell functions, we use double precision to limit floating
// point arithmetic issues
__host__ __device__ real calcNewellNxx(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNyy(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNzz(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNxy(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNxz(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNyx(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNyz(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNzx(int3 idx, real3 cellsize);
__host__ __device__ real calcNewellNzy(int3 idx, real3 cellsize);