#include "ferromagnet.hpp"
#include "field.hpp"
#include "torque.hpp"

Torque::Torque(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "torque", "T") {}

__global__ void k_torque(CuField* torque,
                         const CuField* mField,
                         const CuField* hField,
                         real alpha) {
  if (!torque->cellInGrid())
    return;
  real3 m = mField->cellVector();
  real3 h = hField->cellVector();
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  real3 t = -mxh - alpha * mxmxh;
  torque->setCellVector(t);
}

void Torque::evalIn(Field* torque) const {
  auto h = ferromagnet_->effectiveField()->eval();
  k_torque<<<1, torque->grid().ncells()>>>(torque->cu(),
                                           ferromagnet_->magnetization()->cu(),
                                           h->cu(), ferromagnet_->alpha);
}
