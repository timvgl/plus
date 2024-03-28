#include "effectivefield.hpp"

#include "anisotropy.hpp"
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
  if (h.ncomp() == 3)
    h += evalDemagField(magnet); //ignore (for now) in case of AFM
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalEffectiveField, comp,
                            "effective_field", "T");
}