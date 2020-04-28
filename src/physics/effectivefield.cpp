#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "world.hpp"

EffectiveField::EffectiveField(Ferromagnet* ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "effective_field", "T") {}

void EffectiveField::evalIn(Field* result) const {
  auto anisField = ferromagnet_->anisotropyField()->eval();

  auto exchField = ferromagnet_->exchangeField()->eval();
  add(result, exchField.get(), anisField.get());

  auto externalField = ferromagnet_->externalField()->eval();
  add(result, result, externalField.get());

  if (ferromagnet_->enableDemag) {
    auto demagField = ferromagnet_->demagField()->eval();
    add(result, result, demagField.get());
  }
}