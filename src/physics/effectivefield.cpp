#include "effectivefield.hpp"

#include "ferromagnet.hpp"
#include "field.hpp"
#include "anisotropy.hpp"
#include "exchange.hpp"
#include "demag.hpp"
#include "zeeman.hpp"

#include"fieldops.hpp"

Field evalEffectiveField(const Ferromagnet *magnet){
  Field h = demagFieldQuantity(magnet)();
  anisotropyFieldQuantity(magnet).addToField(&h);
  exchangeFieldQuantity(magnet).addToField(&h);
  externalFieldQuantity(magnet).addToField(&h);
  return h;
}

FM_FieldQuantity effectiveFieldQuantity(const Ferromagnet * magnet) {
  return FM_FieldQuantity(magnet, evalEffectiveField, 3, "effective_field", "T");
}