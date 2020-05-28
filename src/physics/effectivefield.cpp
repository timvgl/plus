#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "anisotropy.hpp"
#include "exchange.hpp"
#include "demag.hpp"
#include "zeeman.hpp"

EffectiveField::EffectiveField(Handle<Ferromagnet> ferromagnet)
    : FerromagnetFieldQuantity(ferromagnet, 3, "effective_field", "T") {}

void EffectiveField::evalIn(Field* result) const {
  result->makeZero();

  AnisotropyField(ferromagnet_).addTo(result);
  ExchangeField(ferromagnet_).addTo(result);
  ExternalField(ferromagnet_).addTo(result);
  DemagField(ferromagnet_).addTo(result);
}