#include "demag.hpp"

DemagField::DemagField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "demag_field", "T") {}

void DemagField::evalIn(Field* result) const {
}