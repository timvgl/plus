#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"
#include "ncafm.hpp"

Field evalFMFullMag(const Ferromagnet* magnet) {
  return magnet->msat.eval() * magnet->magnetization()->field();
}

Field evalHMFullMag(const HostMagnet* magnet) {
  auto sublattices = magnet->sublattices();
  Field result = Field(sublattices[0]->system(), 3);   

  Field ms0 = sublattices[0]->msat.eval();      
  const Field& m0 = sublattices[0]->magnetization()->field();

  result = ms0 * m0;
  ms0.markLastUse();
  
  for (int i = 1; i < (int)sublattices.size(); ++i) {
    Field msi = sublattices[i]->msat.eval(); 
    const Field& mi = sublattices[i]->magnetization()->field();
    addTo(result, msi, mi);
    msi.markLastUse();
  }
  result.markLastUse();
  return result;
}

Field evalHMFullMagOn(const HostMagnet* magnet, cudaStream_t stream_) {
  auto sublattices = magnet->sublattices();
  Field result = Field(sublattices[0]->system(), 3, stream_);   

  Field ms0 = sublattices[0]->msat.eval(stream_);      
  const Field& m0 = sublattices[0]->magnetization()->field();

  result = ms0 * m0;
  ms0.markLastUse(stream_);
  
  for (int i = 1; i < (int)sublattices.size(); ++i) {
    Field msi = sublattices[i]->msat.eval(stream_); 
    const Field& mi = sublattices[i]->magnetization()->field();
    addTo(result, msi, mi);
    msi.markLastUse();
  }
  result.markLastUse();
  return result;
}

FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet* magnet) {
    return FM_FieldQuantity(magnet, evalFMFullMag, 3, "full_magnetization", "A/m");
}

AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet* magnet) {
    return AFM_FieldQuantity(magnet, evalHMFullMag, 3, "full_magnetization", "A/m");
}

NcAfm_FieldQuantity fullMagnetizationQuantity(const NcAfm* magnet) {
    return NcAfm_FieldQuantity(magnet, evalHMFullMag, 3, "full_magnetization", "A/m");
}