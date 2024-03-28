#include "conductivitytensor.hpp"
#include "cudalaunch.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"

bool conductivityTensorAssuredZero(const Ferromagnet* magnet) {
  return magnet->conductivity.assuredZero();
}

__global__ static void k_conductTensor(CuField conductivity,
                                       const CuParameter conductivity0,
                                       const CuParameter conductivity02,
                                       const CuParameter amrRatio,
                                       const CuParameter amrRatio2,
                                       const CuField mField) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bool afm = (mField.ncomp == 6);

  if (!mField.cellInGeometry(idx))
    return;

  const real amr = amrRatio.valueAt(idx);
  const real amr2 = amrRatio2.valueAt(idx);
  const real c0 = conductivity0.valueAt(idx);
  const real c02 = conductivity02.valueAt(idx);
  const real fac = 6 * amr / (6 + amr);


  if (!afm) {
    const real3 m = mField.FM_vectorAt(idx);
    conductivity.setValueInCell(idx, 0, c0 * (1 - fac * (m.x * m.x - 1. / 3.)));
    conductivity.setValueInCell(idx, 1, c0 * (1 - fac * (m.y * m.y - 1. / 3.)));
    conductivity.setValueInCell(idx, 2, c0 * (1 - fac * (m.z * m.z - 1. / 3.)));
    conductivity.setValueInCell(idx, 3, c0 * fac * m.x * m.y);
    conductivity.setValueInCell(idx, 4, c0 * fac * m.x * m.z);
    conductivity.setValueInCell(idx, 5, c0 * fac * m.y * m.z);
  }
  else {
    const real6 m = mField.AFM_vectorAt(idx);
    const real fac2 = 6 * amr2 / (6 + amr2);
    conductivity.setValueInCell(idx, 0, c0 * (1 - fac * (m.x1 * m.x1 - 1. / 3.)));
    conductivity.setValueInCell(idx, 1, c0 * (1 - fac * (m.y1 * m.y1 - 1. / 3.)));
    conductivity.setValueInCell(idx, 2, c0 * (1 - fac * (m.z1 * m.z1 - 1. / 3.)));
    conductivity.setValueInCell(idx, 3, c0 * fac * m.x1 * m.y1);
    conductivity.setValueInCell(idx, 4, c0 * fac * m.x1 * m.z1);
    conductivity.setValueInCell(idx, 5, c0 * fac * m.y1 * m.z1);
    conductivity.setValueInCell(idx, 6, c0 * (1 - fac2 * (m.x2 * m.x2 - 1. / 3.)));
    conductivity.setValueInCell(idx, 7, c0 * (1 - fac2 * (m.y2 * m.y2 - 1. / 3.)));
    conductivity.setValueInCell(idx, 8, c0 * (1 - fac2 * (m.z2 * m.z2 - 1. / 3.)));
    conductivity.setValueInCell(idx, 9, c02 * fac2 * m.x2 * m.y2);
    conductivity.setValueInCell(idx, 10, c02 * fac2 * m.x2 * m.z2);
    conductivity.setValueInCell(idx, 11, c02 * fac2 * m.y2 * m.z2); 
  }
}

Field evalConductivityTensor(const Ferromagnet* magnet) {
  Field conductivity(magnet->system(), 6);
  int ncells = magnet->grid().ncells();
  auto conduct0 = magnet->conductivity.cu();
  auto conduct02 = magnet->conductivity2.cu();
  auto amr = magnet->amrRatio.cu();
  auto amr2 = magnet->amrRatio2.cu();
  auto mField = magnet->magnetization()->field().cu();
  cudaLaunch(ncells, k_conductTensor, conductivity.cu(), conduct0, conduct02, amr, amr2, mField);
  return conductivity;
}

FM_FieldQuantity conductivityTensorQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->field().cu().ncomp;
  return FM_FieldQuantity(magnet, evalConductivityTensor, comp,
                          "conductivity_tensor", "T");
}
