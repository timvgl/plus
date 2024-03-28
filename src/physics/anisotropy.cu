#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include <stdio.h>

bool unianisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->ku1.assuredZero() && magnet->ku2.assuredZero()
       && magnet->ku12.assuredZero() && magnet->ku22.assuredZero())
       || magnet->anisU.assuredZero()
       || (magnet->msat.assuredZero() && magnet->msat2.assuredZero());
}

bool cubicanisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->kc1.assuredZero() && magnet->kc2.assuredZero() && magnet->kc3.assuredZero()
       && magnet->kc12.assuredZero() && magnet->kc22.assuredZero() && magnet->kc32.assuredZero())
       || (magnet->anisC1.assuredZero() && magnet->anisC2.assuredZero())
       || (magnet->msat.assuredZero() && magnet->msat2.assuredZero());
}

__global__ void k_unianisotropyField(CuField hField,
                                  const CuField mField,
                                  const FM_CuVectorParameter anisU,
                                  const CuParameter FM_Ku1,
                                  const CuParameter AFM_Ku1,
                                  const CuParameter FM_Ku2,
                                  const CuParameter AFM_Ku2,
                                  const CuParameter msat,
                                  const CuParameter msat2,
                                  const int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      if (comp == 3) {
        hField.setVectorInCell(idx, real3{0, 0, 0});
      }
      else if (comp == 6) {
        hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
      }
    return;
  }

  if (comp == 3 && msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
  else if (comp == 6 && msat.valueAt(idx) == 0. && msat2.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
      return;
    }

  if (comp == 3) {
    real3 u = normalized(anisU.FM_vectorAt(idx));
    real3 m = mField.FM_vectorAt(idx);
    real k1 = FM_Ku1.valueAt(idx);
    real k2 = FM_Ku2.valueAt(idx);
    real Ms = msat.valueAt(idx);

    real mu = dot(m, u);

    real3 h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / Ms;
    hField.setVectorInCell(idx, h);
  }

  else if (comp == 6) {
    real3 anis = normalized(anisU.FM_vectorAt(idx));
    real6 u = real6{anis.x, anis.y, anis.z, anis.x, anis.y, anis.z};
    real6 m = mField.AFM_vectorAt(idx);
    real2 k1 = {FM_Ku1.valueAt(idx), AFM_Ku1.valueAt(idx)};
    real2 k2 = {FM_Ku2.valueAt(idx), AFM_Ku2.valueAt(idx)};
    real2 Ms = {msat.valueAt(idx), msat2.valueAt(idx)};

    real2 mu = dot(m, u);

    real6 h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / Ms;
    hField.setVectorInCell(idx, h);
  }
}

__global__ void k_cubicanisotropyField(CuField hField,
                                  const CuField mField,
                                  const FM_CuVectorParameter anisC1,
                                  const FM_CuVectorParameter anisC2,
                                  const CuParameter Kc1,
                                  const CuParameter Kc2,
                                  const CuParameter Kc3,
                                  const CuParameter Kc12,
                                  const CuParameter Kc22,
                                  const CuParameter Kc32,
                                  const CuParameter msat,
                                  const CuParameter msat2,
                                  const int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      if (comp == 3) {
        hField.setVectorInCell(idx, real3{0, 0, 0});
      }
      else if (comp == 6) {
        hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
      }
    return;
  }

  if (comp == 3 && msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }
  else if (comp == 6 && msat.valueAt(idx) == 0. && msat2.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real6{0, 0, 0, 0, 0, 0});
      return;
    }

  real3 c1 = normalized(anisC1.FM_vectorAt(idx));
  real3 c2 = normalized(anisC2.FM_vectorAt(idx));
  real3 c3 = cross(c1, c2);
  
  if (comp == 3) {

    real3 m = mField.FM_vectorAt(idx);

    real kc1 = Kc1.valueAt(idx);
    real kc2 = Kc2.valueAt(idx);
    real kc3 = Kc3.valueAt(idx);

    real Ms = msat.valueAt(idx);

    real c1m = dot(c1, m);
    real c2m = dot(c2, m);
    real c3m = dot(c3, m);

    real3 h = -2 * kc1 * ( (c2m * c2m + c3m * c3m) * (c1m * c1)
              + (c1m * c1m + c3m * c3m) * (c2m * c2)
              + (c1m * c1m + c2m * c2m) * (c3m * c3)) / Ms
              - 2 * kc2 * ((c2m * c2m * c3m * c3m) * (c1m * c1)
              + (c1m * c1m * c3m * c3m) * (c2m * c2)
              + (c1m * c1m * c2m * c2m) * (c3m * c3)) / Ms
              - 4 * kc3 * ((c2m * c2m * c2m * c2m + c3m * c3m * c3m * c3m) * (c1m * c1m * c1m * c1)
              + (c1m * c1m * c1m * c1m + c3m * c3m * c3m * c3m) * (c2m * c2m * c2m * c2)
              + (c1m * c1m * c1m * c1m + c2m * c2m * c2m * c2m) * (c3m * c3m * c3m * c3)) / Ms;
  
    hField.setVectorInCell(idx, h);
  }
  else if (comp == 6) {

    real6 cc1 = real6{c1.x, c1.y, c1.z, c1.x, c1.y, c1.z};
    real6 cc2 = real6{c2.x, c2.y, c2.z, c2.x, c2.y, c2.z};
    real6 cc3 = real6{c3.x, c3.y, c3.z, c3.x, c3.y, c3.z};
    real6 m = mField.AFM_vectorAt(idx);

    real2 kc1 = {Kc1.valueAt(idx), Kc12.valueAt(idx)};
    real2 kc2 = {Kc2.valueAt(idx), Kc22.valueAt(idx)};
    real2 kc3 = {Kc3.valueAt(idx), Kc32.valueAt(idx)};

    real2 Ms = {msat.valueAt(idx), msat2.valueAt(idx)};

    real2 c1m = dot(cc1, m);
    real2 c2m = dot(cc2, m);
    real2 c3m = dot(cc3, m);

    real6 h = -2 * kc1 * ( (c2m * c2m + c3m * c3m) * (c1m * cc1)
              + (c1m * c1m + c3m * c3m) * (c2m * cc2)
              + (c1m * c1m + c2m * c2m) * (c3m * cc3)) / Ms
              - 2 * kc2 * ((c2m * c2m * c3m * c3m) * (c1m * cc1)
              + (c1m * c1m * c3m * c3m) * (c2m * cc2)
              + (c1m * c1m * c2m * c2m) * (c3m * cc3)) / Ms
              - 4 * kc3 * ((c2m * c2m * c2m * c2m + c3m * c3m * c3m * c3m) * (c1m * c1m * c1m * cc1)
              + (c1m * c1m * c1m * c1m + c3m * c3m * c3m * c3m) * (c2m * c2m * c2m * cc2)
              + (c1m * c1m * c1m * c1m + c2m * c2m * c2m * c2m) * (c3m * c3m * c3m * cc3)) / Ms;

    hField.setVectorInCell(idx, h);
  }
}


Field evalAnisotropyField(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();
  Field result(magnet->system(), comp);
  
  if (unianisotropyAssuredZero(magnet) && cubicanisotropyAssuredZero(magnet)) {
    result.makeZero();
    return result;
  }

  CuField h = result.cu();
  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto msat2 = magnet->msat2.cu();
  int ncells = magnet->grid().ncells();

  if (!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku12 = magnet->ku12.cu();
    auto ku2 = magnet->ku2.cu();
    auto ku22 = magnet->ku22.cu();
    cudaLaunch(ncells, k_unianisotropyField, h, m, anisU,
               ku1, ku12, ku2, ku22, msat, msat2, comp);
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet->anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    auto kc12 = magnet->kc12.cu();
    auto kc22 = magnet->kc22.cu();
    auto kc32 = magnet->kc32.cu();
    cudaLaunch(ncells, k_cubicanisotropyField, h, m, anisC1, anisC2,
               kc1, kc2, kc3, kc12, kc22, kc32, msat, msat2, comp);
  }
  return result;
}

__global__ void k_unianisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const FM_CuVectorParameter anisU,
                                          const CuParameter Ku1,
                                          const CuParameter Ku12,
                                          const CuParameter Ku2,
                                          const CuParameter Ku22,
                                          const CuParameter msat,
                                          const CuParameter msat2,
                                          const int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      if (comp == 3)
        edens.setValueInCell(idx, 0, 0.0);
      else if (comp == 6)
        edens.setValueInCell(idx, real2{0.0, 0.0});
    }
    return;
  }

  if (msat.valueAt(idx) == 0.0 && msat2.valueAt(idx) == 0.0) {
    if (comp == 3)
      edens.setValueInCell(idx, 0, 0.0);
    else if (comp == 6)
      edens.setValueInCell(idx, real2{0.0, 0.0});
    return;
  }

  if (comp == 3) {
    real3 u = normalized(anisU.FM_vectorAt(idx));
    real3 m = mField.FM_vectorAt(idx);
    real k1 = Ku1.valueAt(idx);
    real k2 = Ku2.valueAt(idx);

    real mu = dot(m, u);

    real e = 0.0;
    e -= k1 * mu * mu;
    e -= k2 * mu * mu * mu * mu;
    edens.setValueInCell(idx, 0, e);
  }

  else if (comp == 6) {
    real3 anis = normalized(anisU.FM_vectorAt(idx));
    real6 u = real6{anis.x, anis.y, anis.z, anis.x, anis.y, anis.z};
    real6 m = mField.AFM_vectorAt(idx);
    real2 k1 = {Ku1.valueAt(idx), Ku12.valueAt(idx)};
    real2 k2 = {Ku2.valueAt(idx), Ku22.valueAt(idx)};

    real2 mu = dot(m, u);

    real2 e = {0.0, 0.0};
    e -= k1 * mu * mu;
    e -= k2 * mu * mu * mu * mu;
    edens.setValueInCell(idx, e);
  }
}

__global__ void k_cubanisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const FM_CuVectorParameter anisC1,
                                          const FM_CuVectorParameter anisC2,
                                          const CuParameter Kc1,
                                          const CuParameter Kc2,
                                          const CuParameter Kc3,
                                          const CuParameter Kc12,
                                          const CuParameter Kc22,
                                          const CuParameter Kc32,
                                          const CuParameter msat,
                                          const CuParameter msat2,
                                          const int comp) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      if (comp == 3)
        edens.setValueInCell(idx, 0, 0.0);
      else if (comp == 6)
        edens.setValueInCell(idx, real2{0.0, 0.0});
    }
    return;
  }

  if (msat.valueAt(idx) == 0.0 && msat2.valueAt(idx) == 0.0) {
    if (comp == 3)
      edens.setValueInCell(idx, 0, 0.0);
    else if (comp == 6)
      edens.setValueInCell(idx, real2{0.0, 0.0});
    return;
  }

  real3 c1 = normalized(anisC1.FM_vectorAt(idx));
  real3 c2 = normalized(anisC2.FM_vectorAt(idx));
  real3 c3 = cross(c1, c2);

  if (comp == 3) {
    real kc1 = Kc1.valueAt(idx);
    real kc2 = Kc2.valueAt(idx);
    real kc3 = Kc3.valueAt(idx);
    real3 m = mField.FM_vectorAt(idx);

    real c1m = dot(c1, m);
    real c2m = dot(c2, m);
    real c3m = dot(c3, m);
    
    real e = 0.0;
    e += kc1 * (c1m * c1m * c2m * c2m
              + c1m * c1m * c3m * c3m
              + c2m * c2m * c3m * c3m);
    e += kc2 * c1m * c1m * c2m * c2m * c3m * c3m;
    e += kc3 * (c1m * c1m * c1m * c1m * c2m * c2m * c2m * c2m
              + c1m * c1m * c1m * c1m * c3m * c3m * c3m * c3m
              + c2m * c2m * c2m * c2m * c3m * c3m * c3m * c3m);
    edens.setValueInCell(idx, 0, e);
  }

  else if (comp == 6) {
    real6 cc1 = real6{c1.x, c1.y, c1.z, c1.x, c1.y, c1.z};
    real6 cc2 = real6{c2.x, c2.y, c2.z, c2.x, c2.y, c2.z};
    real6 cc3 = real6{c3.x, c3.y, c3.z, c3.x, c3.y, c3.z};
    real6 m = mField.AFM_vectorAt(idx);

    real2 kc1 = {Kc1.valueAt(idx), Kc12.valueAt(idx)};
    real2 kc2 = {Kc2.valueAt(idx), Kc22.valueAt(idx)};
    real2 kc3 = {Kc3.valueAt(idx), Kc32.valueAt(idx)};

    real2 c1m = dot(cc1, m);
    real2 c2m = dot(cc2, m);
    real2 c3m = dot(cc3, m);

    real2 e = {0.0, 0.0};

    e += kc1 * (c1m * c1m * c2m * c2m
              + c1m * c1m * c3m * c3m
              + c2m * c2m * c3m * c3m);
    e += kc2 * c1m * c1m * c2m * c2m * c3m * c3m;
    e += kc3 * (c1m * c1m * c1m * c1m * c2m * c2m * c2m * c2m
              + c1m * c1m * c1m * c1m * c3m * c3m * c3m * c3m
              + c2m * c2m * c2m * c2m * c3m * c3m * c3m * c3m);

    edens.setValueInCell(idx, e);
  }
}

Field evalAnisotropyEnergyDensity(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();
  Field edens(magnet->system(), comp / 3);

  if (unianisotropyAssuredZero(magnet) && cubicanisotropyAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }

  CuField e = edens.cu();

  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto msat2 = magnet->msat2.cu();
  int ncells = magnet->grid().ncells();

  if(!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku12 = magnet->ku12.cu();
    auto ku2 = magnet->ku2.cu();
    auto ku22 = magnet->ku22.cu();
    cudaLaunch(ncells, k_unianisotropyEnergyDensity, e, m,
               anisU, ku1, ku12, ku2, ku22, msat, msat2, comp);
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet-> anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    auto kc12 = magnet->kc12.cu();
    auto kc22 = magnet->kc22.cu();
    auto kc32 = magnet->kc32.cu();
    cudaLaunch(ncells, k_cubanisotropyEnergyDensity, e, m,
               anisC1, anisC2, kc1, kc2, kc3, kc12, kc22, kc32, msat, msat2, comp);
  }
  return edens;
}

real evalAnisotropyEnergy(const Ferromagnet* magnet, const bool sub2) {
  if (unianisotropyAssuredZero(magnet) && cubicanisotropyAssuredZero(magnet))
    return 0;

  real edens;
  if (!sub2) 
    edens = anisotropyEnergyDensityQuantity(magnet).average()[0];
  else 
    edens = anisotropyEnergyDensityQuantity(magnet).average()[1];
  
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalAnisotropyField, comp,
                            "anisotropy_field", "T");  
}

FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet* magnet) {

  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalAnisotropyEnergyDensity, comp / 3,
                            "anisotropy_energy_density", "J/m3");
}

FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet* magnet, const bool sub2) {
  std::string name = sub2 ? "anisotropy_energy2" : "anisotropy_energy";
  return FM_ScalarQuantity(magnet, evalAnisotropyEnergy, sub2, name, "J");
}
