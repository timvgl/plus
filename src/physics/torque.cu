#include "constants.hpp"
#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "torque.hpp"

Torque::Torque(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "torque", "T") {}

__global__ void k_torque(CuField torque,
                         CuField mField,
                         CuField hField,
                         CuParameter alpha) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real a = alpha.valueAt(idx);
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -GAMMALL / (1 + a * a) * (mxh + a * mxmxh);
  torque.setVectorInCell(idx, t);
}

void Torque::evalIn(Field* torque) const {
  auto h = EffectiveField(ferromagnet_).eval();
  auto m = ferromagnet_->magnetization()->field();
  auto alpha = ferromagnet_->alpha.cu();
  int ncells = torque->grid().ncells();
  cudaLaunch(ncells, k_torque, torque->cu(), m->cu(), h.get()->cu(), alpha);
}

RelaxTorque::RelaxTorque(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "damping_torque", "T") {}

__global__ void k_dampingtorque(CuField torque,
                                CuField mField,
                                CuField hField) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!torque.cellInGrid(idx))
    return;
  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real3 t = -GAMMALL * cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

void RelaxTorque::evalIn(Field* torque) const {
  auto h = EffectiveField(ferromagnet_).eval();
  auto m = ferromagnet_->magnetization()->field();
  int ncells = torque->grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque->cu(), m->cu(), h.get()->cu());
}
