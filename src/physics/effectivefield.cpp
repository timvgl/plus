#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"

EffectiveField::EffectiveField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "effective_field", "T") {}

void EffectiveField::evalIn(Field* result) const {
  auto anisField = ferromagnet_->anisotropyField()->eval();
  auto exchField = ferromagnet_->exchangeField()->eval();
  add(result, exchField.get(), anisField.get());
}