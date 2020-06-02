#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "interfacialdmi.hpp"
#include "parameter.hpp"
#include "world.hpp"

// At the moment we only suport interfacially induced dmi in the xy plane
const real3 interfaceNormal{0,0,1};

bool interfacialDmiAssuredZero(const Ferromagnet* magnet) {
  return magnet->idmi.assuredZero();
}

__device__ static inline real harmonicMean(real a, real b) {
  if (a + b == 0.0)
    return 0.0;
  return 2 * a * b / (a + b);
}

__global__ void k_interfacialDmiField(CuField hField,
                                      const CuField mField,
                                      const CuParameter idmi,
                                      const CuParameter msat,
                                      real3 interfaceNormal,
                                      real3 cellsize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!hField.cellInGrid(idx))
    return;

  int3 coo = hField.grid.index2coord(idx);

  real3 m = mField.vectorAt(idx);
  real d = idmi.valueAt(idx) / msat.valueAt(idx);

  real3 h{0, 0, 0};  // accumulate exchange field in cell idx

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

  for (int3 relcoo : neighborRelativeCoordinates) {
    int3 coo_ = coo + relcoo;
    int idx_ = hField.grid.coord2index(coo_);

    if (hField.cellInGrid(coo_)) {
      real cs =
          abs(cellsize.x * relcoo.x + cellsize.y * relcoo.y +
              cellsize.z * relcoo.z);  // cellsize in direction of the neighbor

      real3 m_ = mField.vectorAt(idx_);
      real d_ = idmi.valueAt(idx_) / msat.valueAt(idx_);
      real3 dr{(real)relcoo.x, (real)relcoo.y, (real)relcoo.z}; // unit vector from cell to neighbor

      real3 dmivec = harmonicMean(d, d_) * cross(dr, interfaceNormal);
      h += cross(dmivec, m_) / cs;
    }
  }

  hField.setVectorInCell(idx, h);
}

Field evalInterfacialDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->grid(), 3);
  if (interfacialDmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }
  cudaLaunch(hField.grid().ncells(), k_interfacialDmiField, hField.cu(),
             magnet->magnetization()->field().cu(), magnet->idmi.cu(),
             magnet->msat.cu(), interfaceNormal, magnet->world()->cellsize());
  return hField;
}

__global__ void k_interfacialDmiEnergyDensity(CuField edens,
                                              const CuField mag,
                                              const CuField hfield,
                                              const CuParameter msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!edens.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real3 h = hfield.vectorAt(idx);
  real3 m = mag.vectorAt(idx);

  edens.setValueInCell(idx, 0, -0.5 * Ms * dot(m, h));
}

Field evalInterfacialDmiEnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->grid(), 1);
  if (interfacialDmiAssuredZero(magnet)) {
    edens.makeZero();
    return edens;
  }
  Field h = evalInterfacialDmiField(magnet);
  cudaLaunch(edens.grid().ncells(), k_interfacialDmiEnergyDensity, edens.cu(),
             magnet->magnetization()->field().cu(), h.cu(), magnet->msat.cu());
  return edens;
}

real evalInterfacialDmiEnergy(const Ferromagnet* magnet) {
  if (interfacialDmiAssuredZero(magnet))
    return 0;
  real edens = interfacialDmiEnergyDensityQuantity(magnet).average()[0];
  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity interfacialDmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInterfacialDmiField, 3,
                          "interfacialdmi_field", "T");
}

FM_FieldQuantity interfacialDmiEnergyDensityQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalInterfacialDmiEnergyDensity, 1,
                          "interfacialdmi_emergy_density", "J/m3");
}

FM_ScalarQuantity interfacialDmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalInterfacialDmiEnergy,
                           "interfacialdmi_energy", "J");
}
