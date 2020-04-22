#include "anisotropy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "cudalaunch.hpp"
#include "parameter.hpp"

AnisotropyField::AnisotropyField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "anisotropy_field", "T") {}

__global__ void k_anisotropyField(CuField hField,
                                  const CuField mField,
                                  real3 anisU,
                                  CuParameter Ku1,
                                  real msat) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGrid(idx))
    return;
  real3 u = normalized(anisU);
  real3 m = mField.vectorAt(idx);
  real k = Ku1.valueAt(idx);
  real3 h = 2 * k * dot(m, u) * u / msat;
  hField.setVectorInCell(idx,h);
}

void anisotropyField(Field* hField, const Ferromagnet* ferromagnet) {
  CuField h = hField->cu();
  const CuField m = ferromagnet->magnetization()->field()->cu();
  real3 anisU = ferromagnet->anisU;
  //real ku1 = ferromagnet->ku1;
  auto ku1 = ferromagnet->ku1.cu();
  real msat = ferromagnet->msat;
  int ncells = hField->grid().ncells();
  cudaLaunch(ncells, k_anisotropyField, h, m, anisU, ku1, msat);
}

void AnisotropyField::evalIn(Field* result) const {
  anisotropyField(result, ferromagnet_);
}