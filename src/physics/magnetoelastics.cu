// TODO: check if these includes are really all necessary
#include "magnetoelastics.hpp"
#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "reduce.hpp"
#include "world.hpp"


bool elasticForceAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->getEnableElastodynamics()) ||
          (magnet->c11.assuredZero() && magnet->c12.assuredZero() &&
           magnet->c44.assuredZero()));
}

bool magnetoelasticAssuredZero(const Ferromagnet* magnet) {
  return ((!magnet->getEnableElastodynamics()) ||
          (magnet->msat.assuredZero()) ||
          (magnet->B1.assuredZero() && magnet->B2.assuredZero()));
}


// TODO:
// see https://github.com/Fredericvdv/Magnetoelasticity_MuMax3/blob/magnetoelastic/cuda/magnetoelasticfield.cu
Field evalMagnetoelasticField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_magnetoelasticField, properties);
  return hField;
}

// TODO:
Field evalElasticForce2D(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (elasticForceAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_elasticForce2D, properties);
  return fField;
}

// TODO:
Field evalElasticForce3D(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (elasticForceAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_elasticForce3D, properties);
  return fField;
}

// TODO:
// see https://github.com/Fredericvdv/Magnetoelasticity_MuMax3/blob/magnetoelastic/cuda/magnetoelasticforce.cu
Field evalMagnetoelasticForce(const Ferromagnet* magnet) {
  Field fField(magnet->system(), 3);
  if (magnetoelasticAssuredZero(magnet)) {
    fField.makeZero();
    return fField;
  }

  // TODO:
  // auto property = magnet->property;
  // cudalaunch(k_magnetoelasticForce, properties);
  return fField;
}


Field evalElasticAcceleration(const Ferromagnet* magnet) {
  // TODO: reevaluate if all lines are needed
  Field aField(magnet->system(), 3, 0.0);

  // TODO: single kernel of additive? probably additive...
  // TODO: what about 1/rho?
  /* // use effective field as an example
  Field h(magnet->system(), 3, 0.0);
  if (!exchangeAssuredZero(magnet)) {h += evalExchangeField(magnet);}
  if (!anisotropyAssuredZero(magnet)) {h += evalAnisotropyField(magnet);}
  if (!externalFieldAssuredZero(magnet)) {h += evalExternalField(magnet);}
  if (!dmiAssuredZero(magnet)) {h += evalDmiField(magnet);}
  if (!demagFieldAssuredZero(magnet)) {h += evalDemagField(magnet);}
  if (magnet->isSublattice())
      if (!inHomoAfmExchangeAssuredZero(magnet)) {h += evalInHomogeneousAfmExchangeField(magnet);}
      if (!homoAfmExchangeAssuredZero(magnet)) {h += evalHomogeneousAfmExchangeField(magnet);}
  return h;
  */

  return aField;
}


FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticField, 3,
                          "magnetoelastic_field", "T");
}

FM_FieldQuantity elasticForceQuantity(const Ferromagnet* magnet) {
  if (magnet->grid().size().z == 1)  // 2D
    return FM_FieldQuantity(magnet, evalElasticForce2D, 3,
                            "elastic_force", "N/m3");
  return FM_FieldQuantity(magnet, evalElasticForce3D, 3, "elastic_force", "N/m3");
}

FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalMagnetoelasticForce, 3,
                          "magnetoelastic_force", "N/m3");
}

FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalElasticAcceleration, 3,
                          "elastic_acceleration", "m/s2");
}
