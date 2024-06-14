#include <memory>

#include "antiferromagnet.hpp"
#include "constants.hpp"
#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "torque.hpp"

Field evalTorque(const Ferromagnet* magnet) {
  Field torque = evalLlgTorque(magnet);
  if (!spinTransferTorqueAssuredZero(magnet))
    torque += evalSpinTransferTorque(magnet);
  return torque;
}

Field evalAFMTorque(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  Field torque = evalAFMLlgTorque(magnet, sublattice);
  if (!spinTransferTorqueAssuredZero(sublattice))
    torque += evalSpinTransferTorque(sublattice);
  return torque;
}
__global__ void k_llgtorque(CuField torque,
                            const CuField mField,
                            const CuField hField,
                            const CuParameter alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.FM_vectorAt(idx);
  real3 h = hField.FM_vectorAt(idx);
  real a = alpha.valueAt(idx);
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -GAMMALL / (1 + a * a) * (mxh + a * mxmxh);
  torque.setVectorInCell(idx, t);

}

Field evalLlgTorque(const Ferromagnet* magnet) {
  const Field& m = magnet->magnetization()->field();
  Field torque(magnet->system(), 3);
  Field h = evalEffectiveField(magnet);
  const Parameter& alpha = magnet->alpha;
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_llgtorque, torque.cu(), m.cu(), h.cu(), alpha.cu());
  return torque;
}

Field evalAFMLlgTorque(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  const Field& s = sublattice->magnetization()->field();
  Field torque(sublattice->system(), 3);
  Field h = evalAFMEffectiveField(magnet, sublattice);
  const Parameter& alpha = sublattice->alpha;
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_llgtorque, torque.cu(), s.cu(), h.cu(), alpha.cu());
  return torque;
}
__global__ void k_dampingtorque(CuField torque,
                                const CuField mField,
                                const CuField hField) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.FM_vectorAt(idx);
  real3 h = hField.FM_vectorAt(idx);
  real3 t = -GAMMALL * cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

Field evalRelaxTorque(const Ferromagnet* magnet) {
  const Field& m = magnet->magnetization()->field();
  Field torque(magnet->system(), 3);
  Field h = evalEffectiveField(magnet);
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque.cu(), m.cu(), h.cu());
  return torque;
}

Field evalAFMRelaxTorque(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  const Field& s = sublattice->magnetization()->field();
  Field torque(sublattice->system(), 3);
  Field h = evalAFMEffectiveField(magnet, sublattice);
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque.cu(), s.cu(), h.cu());
  return torque;
}

FM_FieldQuantity torqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTorque, 3, "torque", "1/s");
}

AFM_FieldQuantity AFM_torqueQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  return AFM_FieldQuantity(magnet, sublattice, evalAFMTorque, 3, "torque", "1/s");
}

FM_FieldQuantity llgTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalLlgTorque, 3, "llg_torque", "1/s");
}

AFM_FieldQuantity AFM_llgTorqueQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  return AFM_FieldQuantity(magnet, sublattice, evalAFMLlgTorque, 3, "llg_torque", "1/s");
}

FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalRelaxTorque, 3, "damping_torque", "1/s");
}

AFM_FieldQuantity AFM_relaxTorqueQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  return AFM_FieldQuantity(magnet, sublattice, evalAFMRelaxTorque, 3, "damping_torque", "1/s");
}