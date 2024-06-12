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
  return h;
}

Field evalAFMEffectiveField(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  Field h = evalAnisotropyField(sublattice);
  h += evalExchangeField(sublattice);
  h += evalExternalField(sublattice);
  h += evalDmiField(sublattice);
  h += evalAFMExchangeField(magnet, sublattice);
  //ignore demag (for now) in case of AFM*/
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet* magnet) {
  int comp = magnet->magnetization()->ncomp();
  return FM_FieldQuantity(magnet, evalEffectiveField, comp,
                            "effective_field", "T");
}

AFM_FieldQuantity AFM_effectiveFieldQuantity(const Antiferromagnet* magnet, const Ferromagnet* sublattice) {
  int comp = sublattice->magnetization()->ncomp();
  return AFM_FieldQuantity(magnet, sublattice, evalAFMEffectiveField, comp,
                            "effective_field", "T");
}