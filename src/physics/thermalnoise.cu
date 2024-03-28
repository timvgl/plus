#include <curand.h>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "thermalnoise.hpp"
#include "world.hpp"

#if FP_PRECISION == SINGLE
const auto& generateRandNormal = curandGenerateNormal;
#elif FP_PRECISION == DOUBLE
const auto& generateRandNormal = curandGenerateNormalDouble;
#endif

bool thermalNoiseAssuredZero(const Ferromagnet* magnet) {
  return magnet->temperature.assuredZero();
}

__global__ void k_thermalNoise(CuField noiseField,
                               const CuParameter msat,
                               const CuParameter msat2,
                               const CuParameter alpha,
                               const CuParameter temperature,
                               real preFactor,
                               int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!noiseField.cellInGeometry(idx)) {
    if (noiseField.cellInGrid(idx)) {
      if (comp == 3)
        noiseField.setVectorInCell(idx, real3{0, 0, 0});
      else if (comp == 6)
        noiseField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
    }
    return;
  }

  if (!noiseField.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real Ms2 = msat2.valueAt(idx);
  real T = temperature.valueAt(idx);
  real a = alpha.valueAt(idx);
  if (comp == 3) {
    real3 noise = noiseField.FM_vectorAt(idx);
    noise *= sqrt(preFactor * a * T / ((1 + a * a) * Ms));
    noiseField.setVectorInCell(idx, noise);
  }
  else if (comp == 6) {
    real6 noise = noiseField.AFM_vectorAt(idx);
    noise *= sqrt(preFactor * a * T / ((1 + a * a) * real2{Ms, Ms2}));
    noiseField.setVectorInCell(idx, noise);
  }
}

Field evalThermalNoise(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  Field noise(magnet->system(), comp);

  if (thermalNoiseAssuredZero(magnet)) {
    noise.makeZero();
    return noise;
  }
  
  int N = noise.grid().ncells();
  real mean = 0.0;
  real stddev = 1.0;
  for (int c = 0; c < comp; c++) {
    generateRandNormal(magnet->randomGenerator, noise.device_ptr(c), N, mean,
                       stddev);
  }

  auto msat = magnet->msat.cu();
  auto msat2 = magnet->msat2.cu();
  auto alpha = magnet->alpha.cu();
  auto temperature = magnet->temperature.cu();
  real cellVolume = magnet->world()->cellVolume();
  real preFactor = 2 * KB * GAMMALL / cellVolume;
  cudaLaunch(N, k_thermalNoise, noise.cu(), msat, msat2, alpha, temperature,
             preFactor, comp);
  return noise;
}

FM_FieldQuantity thermalNoiseQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalThermalNoise, comp, "thermalNoise", "");
}
