#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"

EffectiveField::EffectiveField(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "effective_field", "T") {}

void EffectiveField::evalIn(Field* result) const {
  result->makeZero();

  ferromagnet_->anisotropyField()->addTo(result);
  ferromagnet_->exchangeField()->addTo(result);
  ferromagnet_->externalField()->addTo(result);
  ferromagnet_->demagField()->addTo(result);
}