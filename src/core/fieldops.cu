#include "field.hpp"
#include "fieldops.hpp"

__global__ void k_addFields(CuField* y,
                            real a1,
                            const CuField* x1,
                            real a2,
                            const CuField* x2) {
  if (!y->cellInGrid())
    return;
  int nComp = y->nComponents();
  for (int c = -0; c < nComp; c++) {
    real term1 = a1 * x1->cellValue(c);
    real term2 = a2 * x2->cellValue(c);
    y->setCellValue(c, term1 + term2);
  }
}

// TODO: throw error if grids or number of components do not match
void add(Field* y, real a1, const Field* x1, real a2, const Field* x2) {
  k_addFields<<<1, y->grid().ncells()>>>(y->cu(), a1, x1->cu(), a2, x2->cu());
}

void add(Field* y, const Field* x1, const Field* x2) {
  add(y, 1, x1, 1, x2);
}