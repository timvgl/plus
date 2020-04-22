#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "world.hpp"

EffectiveField::EffectiveField(Ferromagnet* ferromagnet)
    : FerromagnetQuantity(ferromagnet, 3, "effective_field", "T") {}

void EffectiveField::evalIn(Field* result) const {
  auto anisField = ferromagnet_->anisotropyField()->eval();

  auto exchField = ferromagnet_->exchangeField()->eval();
  add(result, exchField.get(), anisField.get());

  if (ferromagnet_->enableDemag) {
    auto demagField = ferromagnet_->demagField()->eval();
    add(result, result, demagField.get());
  }

  real3 b_ext = ferromagnet_->world()->biasMagneticField;
  if (norm(b_ext) != 0) {
    addConstant(result, result, b_ext.x, 0);
    addConstant(result, result, b_ext.y, 1);
    addConstant(result, result, b_ext.z, 2);
  }
}