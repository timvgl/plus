#include "effectivefield.hpp"

#include "afmexchange.hpp"
#include "anisotropy.hpp"
#include "antiferromagnet.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "exchange.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "zeeman.hpp"

Field evalEffectiveField(const Ferromagnet* magnet) {
  Field h = evalAnisotropyField(magnet);
  h += evalExchangeField(magnet);
  h += evalExternalField(magnet);
  h += evalDmiField(magnet);
  h += evalDemagField(magnet);
  if (magnet->isSublattice())
      h += evalAFMExchangeField(magnet);
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
