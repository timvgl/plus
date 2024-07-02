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
  // there will probably be exchange and need to construct a Field anyway
  Field h = evalExchangeField(magnet);
  if (!anisotropyAssuredZero(magnet)) {h += evalAnisotropyField(magnet);}
  if (!externalFieldAssuredZero(magnet)) {h += evalExternalField(magnet);}
  if (!dmiAssuredZero(magnet)) {h += evalDmiField(magnet);}
  if (!demagFieldAssuredZero(magnet)) {h += evalDemagField(magnet);}
  if (magnet->isSublattice())
      if (!afmExchangeAssuredZero(magnet)) {h += evalAFMExchangeField(magnet);}
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}
