#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "world.hpp"

ExchangeField::ExchangeField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "exchange_field", "T") {}

__global__ void k_exchangeField(CuField* hField,
                                const CuField* mField,
                                real aex,
                                real3 cellsize) {
  if (!hField->cellInGrid())
    return;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int3 i = hField->grid().idx2coo(idx);

  real3 m = mField->cellVector();
  real3 h{0, 0, 0};

  int3 neighborRelativeCoordinates[6] = {int3{-1, 0, 0}, int3{0, -1, 0},
                                         int3{0, 0, -1}, int3{1, 0, 0},
                                         int3{0, 1, 0},  int3{0, 0, 1}};

  for (int3 relcoo : neighborRelativeCoordinates) {
    int3 i_ = i + relcoo;
    real dr =
        cellsize.x * relcoo.x + cellsize.y * relcoo.y + cellsize.z * relcoo.z;
    if (hField->cellInGrid(i_)) {
      real3 m_ = mField->cellVector(i_);
      h += aex * m_ / (dr * dr);
    }
  }

  hField->setCellVector(h);
}

void exchangeField(Field* hField, const Ferromagnet* ferromagnet) {
  k_exchangeField<<<1, hField->grid().ncells()>>>(
      hField->cu(), ferromagnet->magnetization()->field()->cu(), ferromagnet->aex,
      ferromagnet->world()->cellsize());
}

void ExchangeField::evalIn(Field* result) const {
  exchangeField(result, ferromagnet_);
}