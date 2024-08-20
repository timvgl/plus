#pragma once

#include "ferromagnetquantity.hpp"

class Ferromagnet;
class Field;

// TODO: this file might be too large, split up into several concepts if needed

// TODO: novel quantities that need to be added somewhere (probably)
// normStrain
// shearStrain
// normStress
// shearStress
// poynting
// B_mel
// F_mel

// TODO: excitations that need to be added somewhere (probably)
// force_density
// exx
// eyy
// ezz
// exz
// exy
// eyz

// TODO: energies
// Edens_el
// E_el
// Edens_mel  // TODO: magnetic, elastic, or both?
// E_mel
// Edens_kin
// E_kin


// layout is usually (TODO: remove)
// assuredZero (bool)
// evalField (Field)
// evalEnergy (real)
// FM_FieldQuantity
// FM_ScalarQuantity


bool elasticForceAssuredZero(const Ferromagnet*);
bool magnetoelasticAssuredZero(const Ferromagnet*);


Field evalMagnetoelasticField(const Ferromagnet*);

Field evalElasticForce2D(const Ferromagnet*);  // TODO: best to make 2D and 3D?
Field evalElasticForce3D(const Ferromagnet*);
Field evalMagnetoelasticForce(const Ferromagnet*);
Field evalElasticAcceleration(const Ferromagnet*);


// Magnetoelastic effective field due to effects of inverse magnetostriction
FM_FieldQuantity magnetoelasticFieldQuantity(const Ferromagnet*);

// Elastic body force due to mechanical stress gradients f = ∇σ = ∇(cε)
FM_FieldQuantity elasticForceQuantity(const Ferromagnet*);
// Magnetoelastic body force due to the magnetostriction effect
FM_FieldQuantity magnetoelasticForceQuantity(const Ferromagnet*);

// Elastic acceleration includes all effects that influence the elastic velocity
// including elastic, magnetoelastic and external body forces, and elastic damping.
FM_FieldQuantity elasticAccelerationQuantity(const Ferromagnet*);
