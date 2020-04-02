#include "anisotropy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "cudalaunch.hpp"

AnisotropyField::AnisotropyField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "anisotropy_field", "T") {}

__global__ void k_anisotropyField(CuField* hField,
                                  const CuField* mField,
                                  real3 anisU,
                                  real Ku1,
                                  real msat) {
  if (!hField->cellInGrid())
    return;
  real3 u = anisU / norm(anisU);
  real3 m = mField->cellVector();
  real3 h = 2 * Ku1 * dot(m, u) * u / msat;
  hField->setCellVector(h);
}

void anisotropyField(Field* hField, const Ferromagnet* ferromagnet) {
  CuField* h = hField->cu();
  const CuField* m = ferromagnet->magnetization()->field()->cu();
  real3 anisU = ferromagnet->anisU;
  real ku1 = ferromagnet->ku1;
  real msat = ferromagnet->msat;
  int ncells = hField->grid().ncells();
  cudaLaunch(ncells, k_anisotropyField, h, m, anisU, ku1, msat);
}

void AnisotropyField::evalIn(Field* result) const {
  anisotropyField(result, ferromagnet_);
}