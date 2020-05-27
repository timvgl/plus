#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "thermalnoise.hpp"
#include "world.hpp"

ThermalNoise::ThermalNoise(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "thermal_noise", "T") {
  curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(generator_, 1234);
}

__global__ void k_thermalNoise(CuField noiseField,
                               CuParameter msat,
                               CuParameter alpha,
                               CuParameter temperature,
                               real preFactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!noiseField.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real T = temperature.valueAt(idx);
  real a = alpha.valueAt(idx);
  real3 noise = noiseField.vectorAt(idx);

  noise *= sqrt(preFactor * a * T / ((1 + a * a) * Ms));

  noiseField.setVectorInCell(idx, noise);
}

ThermalNoise::~ThermalNoise() {
  curandDestroyGenerator(generator_);
}

void ThermalNoise::evalIn(Field* result) const {
  if (assuredZero()) {
    result->makeZero();
    return;
  }

  int N = result->grid().ncells();
  real mean = 0.0;
  real stddev = 1.0;
  for (int c = 0; c < 3; c++) {
    curandGenerateNormal(generator_, result->devptr(c), N, mean, stddev);
    // TODO: make this also work for real = double   (using
    // curandGenerateNormalDouble)
  }

  int ncells = ferromagnet_->grid().ncells();
  auto noise = result->cu();
  auto msat = ferromagnet_->msat.cu();
  auto alpha = ferromagnet_->alpha.cu();
  auto temperature = ferromagnet_->temperature.cu();
  real cellVolume = ferromagnet_->world()->cellVolume();
  real preFactor = 2 * KB * GAMMALL / cellVolume;
  cudaLaunch(ncells, k_thermalNoise, noise, msat, alpha, temperature,
             preFactor);
}

bool ThermalNoise::assuredZero() const {
  return ferromagnet_->temperature.assuredZero();
}