#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

// The homogeneous and non-homogeneous contributions to AFM exchange is split up.
// The homogeneous contribution (considering afmex_cell) corresponds to AFM
// exchange at a single site.
// The non-homogeneous contribution (considering afmex_nn) corresponds to AFM
// exchange between NN cells.

bool nonHomoAfmExchangeAssuredZero(const Ferromagnet*);
bool homoAfmExchangeAssuredZero(const Ferromagnet*);

// Evaluate field
Field evalNonHomogeneousAfmExchangeField(const Ferromagnet*);
Field evalHomogeneousAfmExchangeField(const Ferromagnet*);
// Evaluate energy density
Field evalNonHomoAfmExchangeEnergyDensity(const Ferromagnet*);
Field evalHomoAfmExchangeEnergyDensity(const Ferromagnet*);
// Evaluate energy
real evalNonHomoAfmExchangeEnergy(const Ferromagnet*);
real evalHomoAfmExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity nonHomoAfmExchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity homoAfmExchangeFieldQuantity(const Ferromagnet*);

FM_FieldQuantity nonHomoAfmExchangeEnergyDensityQuantity(const Ferromagnet*);
FM_FieldQuantity homoAfmExchangeEnergyDensityQuantity(const Ferromagnet*);

FM_ScalarQuantity nonHomoAfmExchangeEnergyQuantity(const Ferromagnet*);
FM_ScalarQuantity nonHomoAfmExchangeEnergyQuantity(const Ferromagnet*);

////////////////////////////////////////////////////////////////////////////////////

// returns the deviation from the optimal angle (180°) between magnetization
// vectors in the same cell which are coupled by the intracell exchange interaction.
Field evalAngleField(const Antiferromagnet*);
// The maximal deviation from 180*.
real evalMaxAngle(const Antiferromagnet*);

AFM_FieldQuantity angleFieldQuantity(const Antiferromagnet*);
AFM_ScalarQuantity maxAngle(const Antiferromagnet*);




