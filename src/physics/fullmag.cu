#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"


Field evalFullMag(const Antiferromagnet* magnet) {
  return add(magnet->sub1()->msat.eval(),
             magnet->sub1()->magnetization()->field(),
             magnet->sub2()->msat.eval(),
             magnet->sub2()->magnetization()->field());
}

AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalFullMag, 3, "full_magnetization", "A/m");
}
