#include "anisotropy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"

AnisotropyField::AnisotropyField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "anisotropy_field", "T") {}

__global__ void k_anisotropyField(CuField* hField,
                                  const CuField* mField,
                                  real3 anisU,
                                  real Ku1) {
  if (!hField->cellInGrid())
    return;
  real3 m = mField->cellVector();
  real3 h = Ku1 * anisU * m;
  hField->setCellVector(h);
}

void anisotropyField(Field* hField, const Ferromagnet* ferromagnet) {
  CuField* h = hField->cu();
  const CuField* m = ferromagnet->magnetization()->field()->cu();
  real3 anisU = ferromagnet->anisU;
  real ku1 = ferromagnet->ku1;
  int ncells = hField->grid().ncells();
  k_anisotropyField<<<1, ncells>>>(h, m, anisU, ku1);
}

void AnisotropyField::evalIn(Field* result) const {
  anisotropyField(result, ferromagnet_);
}