#include "anisotropy.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include <stdio.h>

bool unianisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->ku1.assuredZero() && magnet->ku2.assuredZero())
       || magnet->anisU.assuredZero() || magnet->msat.assuredZero();
}

bool cubicanisotropyAssuredZero(const Ferromagnet* magnet) {
  return (magnet->kc1.assuredZero() && magnet->kc2.assuredZero() && magnet->kc3.assuredZero())
       || (magnet->anisC1.assuredZero() && magnet->anisC2.assuredZero())
       || magnet->msat.assuredZero();
}

bool hexanisotropyAssuredZero(const Ferromagnet* magnet) {
    return magnet->khex.assuredZero()
        || magnet->anisCHex.assuredZero()
        || magnet->anisAHex.assuredZero();
}

bool anisotropyAssuredZero(const Ferromagnet* magnet) {
  return (unianisotropyAssuredZero(magnet)
          && cubicanisotropyAssuredZero(magnet)) && hexanisotropyAssuredZero(magnet);
}

__global__ void k_unianisotropyField(CuField hField,
                                  const CuField mField,
                                  const CuVectorParameter anisU,
                                  const CuParameter FM_Ku1,
                                  const CuParameter FM_Ku2,
                                  const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  if (msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k1 = FM_Ku1.valueAt(idx);
  real k2 = FM_Ku2.valueAt(idx);
  real Ms = msat.valueAt(idx);

  real mu = dot(m, u);

  real3 h = (2 * k1 * mu + 4 * k2 * mu * mu * mu) * u / Ms;
  hField.setVectorInCell(idx, h);
}

__global__ void k_cubicanisotropyField(CuField hField,
                                  const CuField mField,
                                  const CuVectorParameter anisC1,
                                  const CuVectorParameter anisC2,
                                  const CuParameter Kc1,
                                  const CuParameter Kc2,
                                  const CuParameter Kc3,
                                  const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  if (msat.valueAt(idx) == 0.) {
      hField.setVectorInCell(idx, real3{0, 0, 0});
      return;
    }

  real3 c1 = normalized(anisC1.vectorAt(idx));
  real3 c2 = normalized(anisC2.vectorAt(idx));
  real3 c3 = cross(c1, c2);
  
  real3 m = mField.vectorAt(idx);

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


__global__ void k_hexagonalAnisotropyField(
    CuField hField,
    const CuField mField,
    const CuVectorParameter anisC,   // Hauptachse c
    const CuVectorParameter anisA,   // Referenzachse a in der Ebene
    const CuParameter Khex,          // 6-fach In-Plane-Anisotropie
    const CuParameter msat)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (!hField.cellInGeometry(idx)) {
        if (hField.cellInGrid(idx)) hField.setVectorInCell(idx, real3{0,0,0});
        return;
    }

    real Ms = msat.valueAt(idx);
    if (Ms == 0.) { hField.setVectorInCell(idx, real3{0,0,0}); return; }

    real khex = Khex.valueAt(idx);

    // --- Orthonormales Dreibein (c, a, b) ---
    real3 c = normalized(anisC.vectorAt(idx));

    real3 a_raw = anisA.vectorAt(idx);
    // a in die Ebene projizieren und normalisieren
    a_raw = a_raw - dot(a_raw, c) * c;
    real3 a = normalized(a_raw);

    // b = c × a, dann normalisieren
    real3 b = normalized(cross(c, a));

    // Fallback, falls Projektion degeneriert (anisA ~ c)
    if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z)) {
        real3 t = (fabs(c.x) < 0.9) ? real3{1,0,0} : real3{0,1,0};
        a = normalized(t - dot(t, c)*c);
        b = normalized(cross(c, a));
    }

    // Magnetisierung + Projektionen
    real3 m  = mField.vectorAt(idx);
    real ma  = dot(m, a);
    real mb  = dot(m, b);

    // Polynome für Ableitungen
    real ma2 = ma*ma, mb2 = mb*mb;
    real ma3 = ma2*ma, mb3 = mb2*mb;
    real ma4 = ma2*ma2, mb4 = mb2*mb2;
    real ma5 = ma4*ma,  mb5 = mb4*mb;

    // dE/d(ma), dE/d(mb) für Re[(ma + i mb)^6] = ma^6 - 15 ma^4 mb^2 + 15 ma^2 mb^4 - mb^6
    real dE_dma =  ma5 - 10.*ma3*mb2 + 5.*ma*mb4;
    real dE_dmb =  5.*ma4*mb - 10.*ma2*mb3 + mb5;

    // H_hex = -(1/Ms) * (dE/dma * a + dE/dmb * b) * 6*Khex
    real3 h_hex = (-6.*khex / Ms) * (dE_dma * a + dE_dmb * b);

    hField.setVectorInCell(idx, h_hex);
}


Field evalAnisotropyField(const Ferromagnet* magnet) {

  Field result(magnet->system(), 3);
  
  if (anisotropyAssuredZero(magnet)) {
    result.makeZero();
    return result;
  }

  CuField h = result.cu();
  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();

  if (!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku2 = magnet->ku2.cu();
    cudaLaunch("anisotropy.cu", ncells, k_unianisotropyField, h, m,
               anisU, ku1, ku2, msat);
    magnet->anisU.markLastUse();
    magnet->ku1.markLastUse();
    magnet->ku2.markLastUse();
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet->anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    cudaLaunch("anisotropy.cu", ncells, k_cubicanisotropyField, h, m,
               anisC1, anisC2, kc1, kc2, kc3, msat);
    magnet->anisC1.markLastUse();
    magnet->anisC2.markLastUse();
    magnet->kc1.markLastUse();
    magnet->kc2.markLastUse();
    magnet->kc3.markLastUse();
  }
  else if (!hexanisotropyAssuredZero(magnet)) {
    auto anisCHex = magnet->anisCHex.cu();
    auto anisAHex = magnet->anisAHex.cu();
    auto khex = magnet->khex.cu();
    cudaLaunch("anisotropy.cu", ncells, k_hexagonalAnisotropyField, h, m,
               anisCHex, anisAHex, khex, msat);
    magnet->anisCHex.markLastUse();
    magnet->anisAHex.markLastUse();
    magnet->khex.markLastUse();
  }
  magnet->msat.markLastUse();
  result.markLastUse();
  return result;
}

__global__ void k_unianisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const CuVectorParameter anisU,
                                          const CuParameter Ku1,
                                          const CuParameter Ku2,
                                          const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      edens.setValueInCell(idx, 0, 0.0);
    }
    return;
  }

  if (msat.valueAt(idx) == 0.0) {
    edens.setValueInCell(idx, 0, 0.0);
    return;
  }

  real3 u = normalized(anisU.vectorAt(idx));
  real3 m = mField.vectorAt(idx);
  real k1 = Ku1.valueAt(idx);
  real k2 = Ku2.valueAt(idx);

  real mu = dot(m, u);

  real e = 0.0;
  e -= k1 * mu * mu;
  e -= k2 * mu * mu * mu * mu;
  edens.setValueInCell(idx, 0, e);
}

__global__ void k_cubanisotropyEnergyDensity(CuField edens,
                                          const CuField mField,
                                          const CuVectorParameter anisC1,
                                          const CuVectorParameter anisC2,
                                          const CuParameter Kc1,
                                          const CuParameter Kc2,
                                          const CuParameter Kc3,
                                          const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx)) {
      edens.setValueInCell(idx, 0, 0.0);
    }
    return;
  }

  if (msat.valueAt(idx) == 0.0) {
    edens.setValueInCell(idx, 0, 0.0);
    return;
  }

  real3 c1 = normalized(anisC1.vectorAt(idx));
  real3 c2 = normalized(anisC2.vectorAt(idx));
  real3 c3 = cross(c1, c2);

  real kc1 = Kc1.valueAt(idx);
  real kc2 = Kc2.valueAt(idx);
  real kc3 = Kc3.valueAt(idx);
  real3 m = mField.vectorAt(idx);

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

Field evalAnisotropyEnergyDensity(const Ferromagnet* magnet) {

  Field edens(magnet->system(), 1);

  if (anisotropyAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }

  CuField e = edens.cu();

  const CuField m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  int ncells = magnet->grid().ncells();

  if(!unianisotropyAssuredZero(magnet)) {
    auto anisU = magnet->anisU.cu();
    auto ku1 = magnet->ku1.cu();
    auto ku2 = magnet->ku2.cu();
    cudaLaunch("anisotropy.cu", ncells, k_unianisotropyEnergyDensity, e, m,
               anisU, ku1, ku2, msat);
    magnet->anisU.markLastUse();
    magnet->ku1.markLastUse();
    magnet->ku2.markLastUse();
  }
  else if(!cubicanisotropyAssuredZero(magnet)) {
    auto anisC1 = magnet->anisC1.cu();
    auto anisC2 = magnet-> anisC2.cu();
    auto kc1 = magnet->kc1.cu();
    auto kc2 = magnet->kc2.cu();
    auto kc3 = magnet->kc3.cu();
    cudaLaunch("anisotropy.cu", ncells, k_cubanisotropyEnergyDensity, e, m,
               anisC1, anisC2, kc1, kc2, kc3, msat);
    magnet->anisC1.markLastUse();
    magnet->anisC2.markLastUse();
    magnet->kc1.markLastUse();
    magnet->kc2.markLastUse();
    magnet->kc3.markLastUse();
  }
  return edens;
}

real evalAnisotropyEnergy(const Ferromagnet* magnet) {
  if (unianisotropyAssuredZero(magnet) && cubicanisotropyAssuredZero(magnet))
    return 0;

  real edens = anisotropyEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity anisotropyFieldQuantity(const Ferromagnet* magnet) {

  return FM_FieldQuantity(magnet, evalAnisotropyField, 3, "anisotropy_field", "T");  
}

FM_FieldQuantity anisotropyEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropyEnergyDensity, 1,
                            "anisotropy_energy_density", "J/m3");
}

FM_ScalarQuantity anisotropyEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAnisotropyEnergy, "anisotropy_energy", "J");
}
