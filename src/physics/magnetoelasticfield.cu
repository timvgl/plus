#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "magnetoelasticfield.hpp"
#include "parameter.hpp"
#include "straintensor.hpp"


bool magnetoelasticAssuredZero(const Ferromagnet* magnet) {
  // use elastodynamics of host if possible
  return (dynamicMagnetoelasticAssuredZero(magnet) &&
          rigidMagnetoelasticAssuredZero(magnet));
}

bool dynamicMagnetoelasticAssuredZero(const Ferromagnet* magnet) {
  // use elastodynamics of host if possible
  bool enableElastodynamics;
  if (magnet->isSublattice()) {
    enableElastodynamics = magnet->hostMagnet()->enableElastodynamics();
  } else {
    enableElastodynamics = magnet->enableElastodynamics();
  }

  return (!enableElastodynamics || magnet->msat.assuredZero() ||
          (magnet->B1.assuredZero() && magnet->B2.assuredZero()));
}

bool rigidMagnetoelasticAssuredZero(const Ferromagnet* magnet) {
  // use rigid strain of host if possible
  bool appliedStrain;
  if (magnet->isSublattice()) {
    appliedStrain = (!magnet->hostMagnet()->rigidNormStrain.assuredZero() ||
                     !magnet->hostMagnet()->rigidShearStrain.assuredZero());
  } else {
    appliedStrain = (!magnet->rigidNormStrain.assuredZero() ||
                     !magnet->rigidShearStrain.assuredZero());
  }

  return (!appliedStrain || magnet->msat.assuredZero() ||
          (magnet->B1.assuredZero() && magnet->B2.assuredZero()));
}

__global__ void k_dynamicMagnetoelasticField(CuField hField,
                                             const CuField mField,
                                             const CuField strain,
                                             const CuParameter B1,
                                             const CuParameter B2,
                                             const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  int ip1, ip2;
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    // If they exceed 3, loop around
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    hField.setValueInCell(idx, i, - 2 / msat.valueAt(idx) *
      (B1.valueAt(idx) *  strain.valueAt(idx, i)       * mField.valueAt(idx, i)   + 
       B2.valueAt(idx) * (strain.valueAt(idx, i+ip1+2) * mField.valueAt(idx, ip1) + 
                          strain.valueAt(idx, i+ip2+2) * mField.valueAt(idx, ip2))));
  }
}

__global__ void k_rigidMagnetoelasticField(CuField hField,
                                           const CuField mField,
                                           const CuVectorParameter normStrain,
                                           const CuVectorParameter shearStrain,
                                           const CuParameter B1,
                                           const CuParameter B2,
                                           const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const CuSystem system = hField.system;
  const Grid grid = system.grid;
  // When outside the geometry, set to zero and return early
  if (!system.inGeometry(idx)) {
    if (grid.cellInGrid(idx)) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
    }
    return;
  }

  int ip1, ip2;
  for (int i=0; i<3; i++) {
    ip1 = i+1; ip2 = i+2;
    // If they exceed 3, loop around
    if (ip1 >= 3) ip1 -= 3;
    if (ip2 >= 3) ip2 -= 3;

    hField.setValueInCell(idx, i, - 2 / msat.valueAt(idx) *
    (B1.valueAt(idx) *  normStrain.valueAt(idx, i)        * mField.valueAt(idx, i)   + 
     B2.valueAt(idx) * (shearStrain.valueAt(idx, i+ip1-1) * mField.valueAt(idx, ip1) + 
                        shearStrain.valueAt(idx, i+ip2-1) * mField.valueAt(idx, ip2))));
  }
}

Field evalMagnetoelasticField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  CuField mField = magnet->magnetization()->field().cu();
  CuParameter B1 = magnet->B1.cu();
  CuParameter B2 = magnet->B2.cu();
  CuParameter msat = magnet->msat.cu();

  if (!rigidMagnetoelasticAssuredZero(magnet)) {  // maybe use rigid strain
    if (magnet->isSublattice()) {  // use strain from host
      CuVectorParameter normStrain = magnet->hostMagnet()->rigidNormStrain.cu();
      CuVectorParameter shearStrain = magnet->hostMagnet()->rigidShearStrain.cu();

      cudaLaunch("magnetoelasticfield.cu", ncells, k_rigidMagnetoelasticField, hField.cu(), mField,
                normStrain, shearStrain, B1, B2, msat);
      magnet->hostMagnet()->rigidNormStrain.markLastUse();
      magnet->hostMagnet()->rigidShearStrain.markLastUse();
    } else {  // independent magnet
      CuVectorParameter normStrain = magnet->rigidNormStrain.cu();
      CuVectorParameter shearStrain = magnet->rigidShearStrain.cu();

      cudaLaunch("magnetoelasticfield.cu", ncells, k_rigidMagnetoelasticField, hField.cu(), mField,
                normStrain, shearStrain, B1, B2, msat);
      magnet->rigidNormStrain.markLastUse();
      magnet->rigidShearStrain.markLastUse();
    }
    magnet->B1.markLastUse();
    magnet->B2.markLastUse();
    magnet->msat.markLastUse();
    hField.markLastUse();
    return hField;
  }

  // otherwise use dynamic strain
  Field strain;
  if (magnet->isSublattice()) {  // use strain from host
    strain = evalStrainTensor(magnet->hostMagnet());
  } else {  // independent magnet
    strain = evalStrainTensor(magnet);
  }

  cudaLaunch("magnetoelasticfield.cu", ncells, k_dynamicMagnetoelasticField, hField.cu(), mField,
            strain.cu(), B1, B2, msat);
  strain.markLastUse();
  magnet->B1.markLastUse();
  magnet->B2.markLastUse();
  magnet->msat.markLastUse();
  hField.markLastUse();
  return hField;
}


Field evalMagnetoelasticEnergyDensity(const Ferromagnet* magnet) {
  if (magnetoelasticAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalMagnetoelasticField(magnet), 0.5);
}

real evalMagnetoelasticEnergy(const Ferromagnet* magnet) {
  if (magnetoelasticAssuredZero(magnet))
    return 0.0;

  real edens = magnetoelasticEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticField, 3,
                          "magnetoelastic_field", "T");
}

FM_FieldQuantity magnetoelasticEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticEnergyDensity, 1,
                          "magnetoelastic_energy_density", "J/m3");
}

FM_ScalarQuantity magnetoelasticEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMagnetoelasticEnergy, "magnetoelastic_energy", "J");
}