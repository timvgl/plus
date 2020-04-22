#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "torque.hpp"
#include "constants.hpp"

Torque::Torque(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "torque", "T") {}

__global__ void k_torque(CuField torque,
                         CuField mField,
                         CuField hField,
                         real alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -GAMMALL / (1 + alpha * alpha) * (mxh + alpha * mxmxh);
  torque.setVectorInCell(idx, t);
}

void Torque::evalIn(Field* torque) const {
  auto h = ferromagnet_->effectiveField()->eval();
  auto m = ferromagnet_->magnetization()->field();
  real alpha = ferromagnet_->alpha;
  int ncells = torque->grid().ncells();
  cudaLaunch(ncells, k_torque, torque->cu(), m->cu(), h.get()->cu(), alpha);
}

RelaxTorque::RelaxTorque(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "damping_torque", "T") {}

__global__ void k_dampingtorque(CuField torque,
                                CuField mField,
                                CuField hField) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real3 t = -GAMMALL*cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

void RelaxTorque::evalIn(Field* torque) const {
  auto h = ferromagnet_->effectiveField()->eval();
  auto m = ferromagnet_->magnetization()->field();
  int ncells = torque->grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque->cu(), m->cu(), h.get()->cu());
}
