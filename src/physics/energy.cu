#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "elasticenergies.hpp"
#include "elastodynamics.hpp"
#include "energy.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "magnetoelasticfield.hpp"
#include "ncafm.hpp"
#include "world.hpp"
#include "zeeman.hpp"

__global__ void k_energyDensity(CuField edens,
                                const CuField mag,
                                const CuField hfield,
                                const CuParameter msat,
                                const real prefactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx))
      edens.setValueInCell(idx, 0, 0.0);
    return;
  }
  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);
  edens.setValueInCell(idx, 0, -prefactor * Ms * dot(m, h));
}

Field evalEnergyDensity(const Ferromagnet* magnet,
                        const Field& h,
                        real prefactor) {
  Field edens(magnet->system(), 1);
  if (magnet->msat.assuredZero()) {
    edens.makeZero();
    edens.markLastUse();
    return edens;
  }

  cudaLaunch("energy.cu", edens.grid().ncells(), k_energyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(),
             magnet->msat.cu(), prefactor);
  magnet->msat.markLastUse();
  edens.markLastUse();
  return edens;
}

real energyFromEnergyDensity(const Magnet* magnet, real edens) {
  int ncells = magnet->system()->cellsInGeo();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * cellVolume * edens;
}

Field evalTotalEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->system(), 1, 0.0);
  Field edemag;
  bool calcDemag = !demagFieldAssuredZero(magnet);
  if (calcDemag) {edemag = evalDemagEnergyDensity(magnet);}
  if (!exchangeAssuredZero(magnet)) {edens += evalExchangeEnergyDensity(magnet);}
  if (!anisotropyAssuredZero(magnet)) {edens += evalAnisotropyEnergyDensity(magnet);}
  if (!externalFieldAssuredZero(magnet)) {edens += evalZeemanEnergyDensity(magnet);}
  if (!inhomoDmiAssuredZero(magnet)) {edens += evalDmiEnergyDensity(magnet);}
  if (!homoAfmExchangeAssuredZero(magnet)) {edens += evalHomoAfmExchangeEnergyDensity(magnet);}
  if (!inHomoAfmExchangeAssuredZero(magnet)) {edens += evalInHomoAfmExchangeEnergyDensity(magnet);}
  
  // magnetoelastics; works if host or if sublattice
  if (!magnetoelasticAssuredZero(magnet)) {edens += evalMagnetoelasticEnergyDensity(magnet);}
  // elastics; only works if independent host
  if (!kineticEnergyAssuredZero(magnet)) {edens += evalKineticEnergyDensity(magnet);}
  if (!elasticityAssuredZero(magnet)) {edens += evalElasticEnergyDensity(magnet);}
  if (calcDemag) {
    fenceStreamToStream(getCudaStreamFFT(), getCudaStream());
    addTo(edens, real{1}, edemag, getCudaStream());
  }
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return edens;
}

Field evalTotalEnergyDensity(const Antiferromagnet* magnet) {
  Field edens = evalTotalEnergyDensity(magnet->sub1()) +
                evalTotalEnergyDensity(magnet->sub2());
  if (!kineticEnergyAssuredZero(magnet)) {edens += evalKineticEnergyDensity(magnet);}
  if (!elasticityAssuredZero(magnet)) {edens += evalElasticEnergyDensity(magnet);}
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return edens;
}

Field evalTotalEnergyDensity(const NcAfm* magnet) {
  Field edens = evalTotalEnergyDensity(magnet->sub1()) +
                evalTotalEnergyDensity(magnet->sub2()) +
                evalTotalEnergyDensity(magnet->sub3());
  if (!kineticEnergyAssuredZero(magnet)) {edens += evalKineticEnergyDensity(magnet);}
  if (!elasticityAssuredZero(magnet)) {edens += evalElasticEnergyDensity(magnet);}
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  return edens;
}

real evalTotalEnergy(const Magnet* magnet) {
  real edensAverage = 0;
  if (const Ferromagnet* mag = magnet->asFM())
    edensAverage = totalEnergyDensityQuantity(mag).average()[0];
  else if (const Antiferromagnet* mag = magnet->asAFM())
    edensAverage = totalEnergyDensityQuantity(mag).average()[0];
  else if (const NcAfm* mag = magnet->asNcAfm())
    edensAverage = totalEnergyDensityQuantity(mag).average()[0];
  else
    throw std::invalid_argument("Cannot calculate energy of instance which "
                                "is no Ferromagnet, Antiferromagnet or"
                                "non-collinear antiferromagnet.");                 
  return energyFromEnergyDensity(magnet, edensAverage);
}

FM_FieldQuantity totalEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet,
    static_cast<Field(*)(const Ferromagnet*)>(evalTotalEnergyDensity),
    1, "total_energy_density", "J/m3");
}

FM_ScalarQuantity totalEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}

AFM_FieldQuantity totalEnergyDensityQuantity(const Antiferromagnet* magnet) {
  return AFM_FieldQuantity(magnet,
    static_cast<Field(*)(const Antiferromagnet*)>(evalTotalEnergyDensity),
    1, "total_energy_density", "J/m3");
}

AFM_ScalarQuantity totalEnergyQuantity(const Antiferromagnet* magnet) {
  return AFM_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}

NcAfm_FieldQuantity totalEnergyDensityQuantity(const NcAfm* magnet) {
  return NcAfm_FieldQuantity(magnet,
    static_cast<Field(*)(const NcAfm*)>(evalTotalEnergyDensity),
    1, "total_energy_density", "J/m3");
}

NcAfm_ScalarQuantity totalEnergyQuantity(const NcAfm* magnet) {
  return NcAfm_ScalarQuantity(magnet, evalTotalEnergy, "total_energy", "J");
}
