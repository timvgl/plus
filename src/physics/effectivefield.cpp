#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "anisotropy.hpp"
#include "exchange.hpp"
#include "demag.hpp"
#include "zeeman.hpp"
#include "interfacialdmi.hpp"

#include"fieldops.hpp"

Field evalEffectiveField(const Ferromagnet *magnet){
  Field h = evalDemagField(magnet);
  h += evalAnisotropyField(magnet);
  h += evalExchangeField(magnet);
  h += evalExternalField(magnet);
  h += evalInterfacialDmiField(magnet);
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}